# Mail Collection and Briefing Delivery

`mail_runtime/` owns persistent reconnaissance mail state. `services/email_bridge.py`
continues to own inbound classification, approval and reply drafts. Its poller may
set IMAP `\Seen`; that flag is never a briefing receipt.

## Read contract

`check_inbox` (`runtime_tools/registry.py` schema, `mail_runtime/inbox.py` handler)
returns external-source-wrapped JSON. Delegated tasks default to `unbriefed_only=true`
unless `unread_only=true`; non-task reads retain ordinary mailbox browsing by default.
Set `unbriefed_only=false` to browse history. The audience is resolved from the
runtime task's `telegram_tasks.user_id`, not model-supplied arguments; a bot-generated
task (`user_id` 0, e.g. a verification auto-retry) resolves to the single
`ALLOWED_USER_IDS` owner because that is where its callback delivers.

"Unbriefed" is bounded: the IMAP search adds `SINCE` now minus
`MAIL_BRIEFING_WINDOW_DAYS` (default 7; 0 disables), reported as
`new_mail_window_days` and per-folder `coverage.since`. Mail older than the window
is history, never new; without this bound every historical message without a
receipt stayed "unbriefed" forever and made coverage numbers unreadable.

The cache identity is `(account hash, folder, UIDVALIDITY, UID)`. The hash covers
configured IMAP host, port and username; it contains no password. Missing
UIDVALIDITY is an explicit failure, never a fallback to unsafe UID-only identity.
Each new message is fetched with BODY.PEEK once and its raw MIME and complete
extracted content are saved in PostgreSQL. Subsequent listings refresh only flags
for cached messages. `mail_id` detail reads paginate saved content without IMAP.
Folders are searched read-only; this path never changes server read flags.

Each result separates `imap_read` and its observation time, collection time,
`body_fully_returned_at`/task ID, current-task full-body coverage
(`body_returned_completely_in_this_task` plus `body_unread_ranges_in_this_task`,
the exact character spans still missing), and `briefing_delivered`. `next` always
asks for a full 12000-character page, whatever the listing page size was. Full-body coverage means text was returned to the agent,
not proof of comprehension. Task-scoped ranges are merged transactionally, so
overlaps cannot conceal gaps. A metadata-only listing does not count as a body read.
The combined message JSON is bounded to 40k characters before recording returned
ranges, below the gateway's 50k cap. Large lists return smaller body pages or
metadata with cached continuation pointers. Display metadata/links are bounded;
complete parsed text and raw MIME remain stored.

Listings return at most 20 messages. Each folder examines up to `limit * 5`
candidates after filtering known deliveries, then results are merged by date.
`coverage` reports eligible/examined/unexamined counts and folder/fetch failures;
`matched_but_not_returned` exposes the combined limit. No result means no returned
matches within this coverage, not proof that an entire mailbox is empty/unchanged.
Unread-only queries explicitly use IMAP UNSEEN. Historical mail without a receipt
is **unrecorded**, not proven new; there is no guessed UID baseline or automatic
backfill from old prose. Moving mail or resetting UIDVALIDITY creates a new identity
and may surface it again; Message-ID deduplication across folders is not inferred.

## Prepare and deliver

Scout has `prepare_mail_briefing(items=[{mail_id, summary}])`, a staged-write tool.
It requires a runtime delegated task, 1–20 distinct cached IDs from this account,
complete body coverage in that task, and 1–1800 character source-attributed summaries.
A coverage rejection names the missing spans and the exact `check_inbox` call to make
next. Batch validation/storage is transactional. Raw mail is already archived by the
read tool; scout should not create duplicate markdown mail archives.

For a non-interrupted, done task without a failed verification verdict, the Telegram
callback sends the prepared summaries themselves, one message per mail, instead of
asking another model to rewrite them. It records each Telegram message ID and
delivery time **after** send success. Per-mail locks and a fresh receipt lookup
avoid concurrent duplicate sends within the Telegram process. A later item failure
leaves earlier successes delivered and later items eligible for the next check.
Failed/interrupted tasks retain prepared items but do not take this delivery path.
Tasks without prepared items retain the normal orchestrator callback. This direct
delivery path does not automatically close a mission.

Verification of a mail-reading task is the ledger itself, by owner decision
(2026-09-18: "don't verify mail checks so strictly"): when a task staged summaries,
recorded body reads, or called `check_inbox` at all, `_run_verification` records a
pass naming the staged mail IDs and read count and skips the LLM verifier; the
reflexion diagnose→revise pass over the report is skipped when the ledger has
entries. The summaries are the deliverable and delivery only follows the verdict,
so a model verifier can never observe it; when it was asked to, every daily run of
2026-09-16..18 failed three times over ("delivery unconfirmed"), sent nothing and
cost up to $0.45 a day. A "no new mail" result is likewise accepted as reported.

A send timeout or crash between Telegram acceptance and receipt storage leaves
delivery unknown and may produce a duplicate on retry; no exactly-once guarantee
is claimed. A receipt means Telegram accepted the summary, not that the user read it.
No auto-send retry worker is added; the next mailbox task can recover unsent mail.

## Storage and deployment

- `mail_briefing_messages`: raw MIME, parsed content, last observed server flags.
- `mail_briefing_reads`: per-task returned body ranges.
- `mail_briefing_items`: per-task/audience summaries and actual send receipts.

Apply `venv/bin/python scripts/schema_migrations.py --only mail-briefing` with
authorized database credentials before deployment. Schema creation is explicit;
runtime tools do not run DDL. State remains in the main PostgreSQL database across
service restarts; there is no automatic mail-history deletion.

Tests: `MAIL_TEST_DATABASE=1 DB_NAME=leninbot_test venv/bin/python -m pytest tests/test_mail_briefing.py`
with test DB credentials. Tests create/drop an isolated schema in the test clone;
IMAP and Telegram are fake. The legacy inbox smoke entrypoint runs this suite.
