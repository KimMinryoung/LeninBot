"""Person-to-history-event links for cards that have none.

The separate link batch was retired on 2026-09-20 and nothing replaced it, so
cards stayed unlinked: 412 of 2,410 people on 2026-09-28, among them
joachim-von-ribbentrop while nazi-soviet-pact existed. The operator restored
linking inside the pipeline tick the same day.

One call per person sees the card and the events that overlap the person's
adult life, and proposes links. A link is written only when its basis is a
verbatim quote of the card, a separate check call (which sees the event's
summary and outcome) finds the quote describes part of THIS event, and the row
passes the shared narrow writer's validation. The check replaced the Jev
citation gate on 2026-09-29: Jev judged whether the quote supported the note,
not whether the note belonged to the event, and passed Hess's 1941 flight to
Scotland for the fall of France. A person for whom the model proposes nothing
gets events=not_applicable (revisited after 180 days). Refused links and
malformed replies go back to the proposer with the reasons in the same tick
(ATTEMPTS proposals at most). What is still refused after that, and errors,
record events=open and are retried after RETRY_HOURS, behind people never tried.
"""
import asyncio
import json
import logging
import re
import uuid

from commulingo.relation_kinds import HISTORY_RELATION_KINDS, definitions_text

from .store import BudgetUnavailable

logger = logging.getLogger(__name__)

FEATURE = 'commulingo_event_person_links'
CHECK_FEATURE = 'commulingo_event_person_link_check'
LANE = 'links'
CHANGED_BY = 'commulingo-pipeline-event-links'
MAX_LINKS = 4
CARD_CHARS = 14_000
SECTION_CHARS = 1_800
SUMMARY_CHARS = 400
EVENT_CONTEXT_CHARS = 1_500  # summary and outcome each, for the check
MIN_BASIS_CHARS = 12
ADULT_AGE = 15
RESERVATION_USD = 0.05
RETRY_HOURS = 6
ATTEMPTS = 3  # first proposal plus two resubmissions in the same tick
CONCURRENCY = 4
# The person page lists each link as one line: relation, then note in small type.
# Stored links sit well under the writer's ceilings (2026-09-28, 3,591 rows:
# relation ko/en median 13/37; note median 57/138) and a model given only the
# ceiling writes to it, so the prompt asks for the median and allows the
# ceiling for complex roles. The writer's own acceptable() enforces the ceiling.
TEXT_CAPS = {'relation_ko': 64, 'relation_en': 164, 'note_ko': 110, 'note_en': 250}

SYSTEM = """You link a person card on a Korean-language history site to the site's history events.

You get the person's card and the events whose years overlap the person's adult life. Propose
a link only when the CARD ITSELF states what this person did or suffered in that event. Your own
knowledge may tell you where to look, but it is never enough: if the card is silent, do not link.
A weak or merely contemporaneous connection is not a link. The card must describe an action
that belongs to THIS event as its title and summary define it: its country, side, place and
dates. A broader war, a neighbouring event or a general career fact is not enough (serving in
the Second World War does not link to a Soviet-front event; commanding the German army does not
link to a Russian revolution; ordering a surrender is not a conference). Use each card quote for
one event only. Most people fit one to three events;
an empty list is a valid answer. At most {max_links} links.

For each link return:
  event_id: one id from the event list, exactly as given.
  kind: one of {kinds}, by what the quote shows the person doing or suffering in this event.
{kind_definitions}
      A historian is linked only when the card names their study of THIS event. An event
      marked before_life ended before the person's adult life: it can only take a historian link.
  relation_ko / relation_en: the person's position in THIS event as a short noun phrase
      (예: 조약에 서명한 외무장관, 진압 지휘, 첫 희생자). Usually about 12 Korean characters /
      35 English; a complex role may run longer, never more than {relation_ko} / {relation_en}.
  note_ko / note_en: ONE short sentence on what they did or what happened to them in this
      event, using only facts the card states. It is a caption under the event title, not a
      summary of the career: one action, with its date or place if the card gives one.
      Usually about 50 Korean characters / 130 English; a complex part may run longer,
      never more than {note_ko} / {note_en}. Count before you answer; a longer link is thrown away.
  basis: an exact, contiguous quote of the card text (either language) that states the
      involvement. Copy it character for character; do not translate, join or shorten
      inside it. 20 to 300 characters.

Writing rules, both languages:
  - No em dash (—) anywhere. Use a comma, a colon, or a new sentence.
  - Korean is 한다체 prose: '~했다', '~이다'.
  - Write 그루지야, never 조지아. Write 조선민주주의인민공화국 or 조선, never 북한.
  - The two languages say the same thing, each as natural prose.

Reply with one JSON object and nothing else:
{{"links": [{{"event_id": "...", "kind": "...", "relation_ko": "...", "relation_en": "...",
  "note_ko": "...", "note_en": "...", "basis": "..."}}], "reason": "one sentence"}}"""


def _years(text):
    found = [int(y) for y in re.findall(r'(?<!\d)(\d{4})(?!\d)', text or '')]
    return (min(found), max(found)) if found else (None, None)


def candidate_events(person, events):
    """Events up to the person's death; all of them when the life is unknown.

    Events that ended before the person's adult life carry before_life=True and
    can only take a historian link. They used to be left out, which made a
    historian of an earlier event impossible to link (2026-09-29: Leonid
    Naumov, b. 1961, a historian of the NKVD, came back "no such event" for the
    Great Terror).
    """
    birth, death = person.get('birth_year'), person.get('death_year')
    if birth is None and death is None:
        birth, death = _years(person.get('years_label'))
    start = birth + ADULT_AGE if birth else None
    end = death or ((birth + 95) if birth else None)
    out = []
    for event in events:
        first, last = _years(event.get('period_label'))
        if first is not None and end is not None and first > end:
            continue
        before = first is not None and start is not None and last < start
        out.append({**event, 'before_life': before})
    return out


def focus_text(event):
    """The side the event centres on, or a marker that it has none (no opponents)."""
    focus = event.get('focus') or {}
    return focus.get('en') or focus.get('ko') or 'none: no single focus; opponents are only those who tried to stop the event itself'


def card_text(person, career, sections):
    parts = [person.get(k) or '' for k in ('epithet_ko', 'epithet_en', 'bio_ko', 'bio_en', 'moment_ko', 'moment_en')]
    parts += [' '.join(filter(None, (c.get('period_label'), c.get('role_ko'), c.get('role_en')))) for c in career]
    for s in sections:
        parts += [s.get('heading_ko') or '', (s.get('body_ko') or '')[:SECTION_CHARS],
                  s.get('heading_en') or '', (s.get('body_en') or '')[:SECTION_CHARS]]
    return '\n'.join(p.strip() for p in parts if p and p.strip())[:CARD_CHARS]


def _norm(text):
    return re.sub(r'\s+', ' ', str(text or '')).strip()


def prompt(person, card, events):
    return json.dumps({
        'person': {'id': person['id'], 'name_ko': person.get('name_ko'), 'name_en': person.get('name_en'),
                   'years': person.get('years_label')},
        'card': card,
        'events': [{'event_id': e['id'], 'period': e.get('period_label'), 'title_ko': e.get('title_ko'),
                    'title_en': e.get('title_en'), 'focus': focus_text(e),
                    **({'before_life': True} if e.get('before_life') else {}),
                    'summary': (e.get('summary_en') or e.get('summary_ko') or '')[:SUMMARY_CHARS]}
                   for e in events],
    }, ensure_ascii=False)


def parse(text):
    raw = str(text or '').strip()
    start, end = raw.find('{'), raw.rfind('}')
    if start == -1 or end < start:
        raise ValueError('no JSON object in reply')
    value = json.loads(raw[start:end + 1])
    links = value.get('links')
    if not isinstance(links, list):
        raise ValueError('reply has no links list')
    return [link for link in links if isinstance(link, dict)], str(value.get('reason') or '')


def screen(links, card, events, acceptable, taken=(), used=()):
    """Split proposals into (kept, rejected-with-reason) before any paid check."""
    ids, text, kept, rejected, seen = {e['id'] for e in events}, _norm(card), [], [], set()
    before = {e['id']: e.get('before_life') for e in events}
    quotes = set(used)
    for link in links[:MAX_LINKS]:
        entry = {k: link.get(k) for k in ('event_id', 'kind', 'relation_ko', 'relation_en', 'note_ko', 'note_en', 'basis')}
        if entry['event_id'] in taken:  # saved by an earlier attempt; a repeat is not a refusal
            continue
        basis = _norm(entry['basis'])
        problem = ('unknown or out-of-period event' if entry['event_id'] not in ids
                   else 'basis too short' if len(basis) < MIN_BASIS_CHARS
                   else 'basis is not a quote of the card' if basis not in text
                   else 'an event before the person\'s adult life can only take a historian link'
                   if before.get(entry['event_id']) and entry['kind'] != 'historian'
                   else 'the same quote is already used for another event' if basis in quotes
                   else acceptable(entry)
                   or ('duplicate event' if entry['event_id'] in seen else ''))
        if problem:
            rejected.append({**entry, 'problem': problem})
        else:
            seen.add(entry['event_id'])
            quotes.add(basis)
            kept.append({**entry, 'basis': basis})
    return kept, rejected


def resubmission(base, previous, rejected, written):
    """The same request plus the previous reply and why its links were refused."""
    problems = '\n'.join(f"- {r.get('event_id') or 'reply'}: {r['problem']}" for r in rejected)
    done = ', '.join(e['event_id'] for e in written) or 'none'
    return (f"{base}\n\nYour previous reply:\n{previous[:6000]}\n\nThese links were refused:\n{problems}\n"
            f"Already saved (do not repeat): {done}.\nReply again with the full JSON object. Fix each refused "
            "link so it passes, or leave it out if the card does not support it.")


CHECK_SYSTEM = """You check proposed links on a Korean-language history site between a person and
one of the site's history events. Each link has the event (title, period, summary, outcome), the
person's proposed kind, role and caption, and a quote from the person's card.

1. belongs: true only when the QUOTE describes something the person did or suffered AS PART OF
THIS EVENT as its summary defines it: the same country or theatre, the same side of it, inside
its period. Answer false when the quote describes:
  - a broader war, era or policy the event is only one part of;
  - a different event near it in time or place (a flight to Britain is not the fall of France;
    a surrender order is not a conference; a Western-front command is not the Eastern front);
  - a general position or career fact that does not name an act in this event;
  - something the caption or role claims but the quote does not say.
For a historian the test is instead whether the quote names their later study or
interpretation of THIS event. An event marked before_life ended before the person's adult
life, so only a historian link can belong to it. Do not use your own knowledge to fill a gap in the quote.
When unsure, answer false.

2. kind: the one kind that fits what the quote shows, whatever was proposed:
""" + definitions_text('  ') + """

Reply with one JSON object and nothing else:
{"verdicts": [{"event_id": "...", "belongs": true, "kind": "...", "reason": "one short sentence"}]}"""


def check_request(person, pairs):
    return json.dumps({
        'person': {'name_en': person.get('name_en'), 'name_ko': person.get('name_ko'),
                   'years': person.get('years_label')},
        'links': [{'event_id': event['id'],
                   'event': {'title_en': event.get('title_en'), 'title_ko': event.get('title_ko'),
                             'period': event.get('period_label'), 'focus': focus_text(event),
                             **({'before_life': True} if event.get('before_life') else {}),
                             'summary': (event.get('summary_en') or event.get('summary_ko') or '')[:EVENT_CONTEXT_CHARS],
                             'outcome': (event.get('outcome_en') or event.get('outcome_ko') or '')[:EVENT_CONTEXT_CHARS]},
                   'proposed_kind': entry['kind'], 'relation_en': entry['relation_en'], 'note_en': entry['note_en'],
                   'quote': entry['basis']} for event, entry in pairs],
    }, ensure_ascii=False)


def parse_verdicts(text):
    raw = str(text or '').strip()
    start, end = raw.find('{'), raw.rfind('}')
    if start == -1 or end < start:
        raise ValueError('no JSON object in check reply')
    verdicts = json.loads(raw[start:end + 1]).get('verdicts')
    if not isinstance(verdicts, list):
        raise ValueError('check reply has no verdicts list')
    return {v.get('event_id'): (v.get('belongs') is True, v.get('kind'), str(v.get('reason') or ''))
            for v in verdicts if isinstance(v, dict)}


class EventLinker:
    def __init__(self, store, *, cap, review_fraction, generate=None, write=None,
                 mark=None, acceptable=None, reservation=RESERVATION_USD):
        self.store, self.cap, self.review_fraction = store, cap, review_fraction
        self.generate, self.write, self.mark = generate, write, mark
        self.acceptable, self.reservation = acceptable, reservation

    # -- data -------------------------------------------------------------
    def people(self, limit, person_ids=None):
        with self.store.transaction() as cur:
            cur.execute('''SELECT p.id FROM commulingo_people p
                LEFT JOIN commulingo_person_enrichment e ON e.person_id=p.id AND e.topic='events'
                WHERE NOT EXISTS (SELECT 1 FROM commulingo_history_event_people l WHERE l.person_id=p.id)
                  AND (e.person_id IS NULL
                       OR (e.status<>'open' AND e.review_after<=now())
                       OR (e.status='open' AND e.updated_at<=now()-%(retry)s*interval '1 hour'))
                  AND NOT EXISTS (SELECT 1 FROM commulingo_agent_suggestions s WHERE s.target_id=p.id
                      AND s.target_type IN ('person','person_section') AND s.status='pending')
                  AND (%(ids)s::text[] IS NULL OR p.id=ANY(%(ids)s))
                -- never-tried people first, then retries oldest first
                ORDER BY e.updated_at NULLS FIRST, p.created_at DESC, p.id LIMIT %(limit)s''',
                {'ids': person_ids, 'limit': limit, 'retry': RETRY_HOURS})
            return [row['id'] for row in cur.fetchall()]

    def load(self, person_id):
        with self.store.transaction() as cur:
            cur.execute('SELECT * FROM commulingo_people WHERE id=%s', (person_id,))
            person = cur.fetchone()
            cur.execute('''SELECT period_label, role_ko, role_en FROM commulingo_person_career_entries
                WHERE person_id=%s ORDER BY sort_order, start_year NULLS LAST''', (person_id,))
            career = cur.fetchall()
            cur.execute('''SELECT heading_ko, heading_en, body_ko, body_en FROM commulingo_person_sections
                WHERE person_id=%s ORDER BY sort_order''', (person_id,))
            sections = cur.fetchall()
            cur.execute('''SELECT id, period_label, title_ko, title_en, summary_ko, summary_en,
                outcome_ko, outcome_en, focus FROM commulingo_history_events ORDER BY sort_order, id''')
            events = cur.fetchall()
        return person, career, sections, events

    # -- one person -------------------------------------------------------
    async def paid(self, system, request, feature):
        token = await asyncio.to_thread(self.store.reserve, self.reservation, lane=LANE, cap=self.cap,
                                        review_fraction=self.review_fraction)
        spent = 0
        try:
            text, spent = await self.generate(system, request, feature)
        finally:
            await asyncio.to_thread(self.store.settle, token, spent)
        return text, spent

    async def link(self, person_id, *, apply=True):
        person, career, sections, events = await asyncio.to_thread(self.load, person_id)
        if not person:
            return {'person': person_id, 'status': 'missing'}
        card = card_text(person, career, sections)
        events = candidate_events(person, events)
        if not card or not events:
            return await self._close(person_id, [], 'no card text or no event in the life span', apply)
        system = SYSTEM.format(max_links=MAX_LINKS, kinds=', '.join(KINDS), kind_definitions=definitions_text(), **TEXT_CAPS)
        base, by_id = prompt(person, card, events), {e['id']: e for e in events}
        text, written, rejected, reason, cost = '', [], [], '', 0
        for attempt in range(ATTEMPTS):
            request = base if attempt == 0 else resubmission(base, text, rejected, written)
            text, spent = await self.paid(system, request, FEATURE)
            cost += spent
            try:
                links, reason = parse(text)
            except ValueError as exc:
                rejected = [{'event_id': None, 'problem': f'reply is not the requested JSON object ({exc})'}]
                continue
            kept, rejected = screen(links, card, events, self.acceptable,
                                    taken={e['event_id'] for e in written},
                                    used={e['basis'] for e in written})
            verdicts = {}
            if kept:
                reply, spent = await self.paid(CHECK_SYSTEM, check_request(
                    person, [(by_id[e['event_id']], e) for e in kept]), CHECK_FEATURE)
                cost += spent
                verdicts = parse_verdicts(reply)  # an unusable check is an error: retried later
            for entry in kept:
                belongs, kind, why = verdicts.get(entry['event_id'], (False, None, 'the check returned no verdict'))
                if not belongs:
                    rejected.append({**entry, 'problem': f'the check found the quote is not part of this event: {why}'})
                    continue
                if kind in KINDS and kind != entry['kind']:
                    rejected.append({**entry, 'problem': f"the check found the kind should be {kind}, "
                                                         f"not {entry['kind']}: {why}. Resubmit with that kind "
                                                         "and a role and caption that fit it"})
                    continue
                if apply:
                    await asyncio.to_thread(self.write, person_id, entry)
                written.append(entry)
            if not rejected:
                break
        # Only the model's own "nothing to link" parks the person (180 days);
        # links still rejected after the resubmissions are retried after RETRY_HOURS.
        status = None if written or not rejected else 'open'
        result = await self._close(person_id, written, reason, apply, status=status, rejected=rejected)
        return {**result, 'rejected': rejected, 'cost_usd': cost, 'attempts': attempt + 1}

    async def _close(self, person_id, written, reason, apply, status=None, rejected=()):
        status = status or ('complete' if written else 'not_applicable')
        note = (f'linked {len(written)} event(s): ' + ', '.join(e['event_id'] for e in written) if written
                else 'retry: ' + '; '.join(f"{r.get('event_id')}: {r['problem']}" for r in rejected)
                if rejected else ('retry: ' if status == 'open' else 'no event the card documents: ')
                + (reason or 'none proposed'))[:500]
        if apply:
            await asyncio.to_thread(self.mark, person_id, status, note)
        return {'person': person_id, 'status': status, 'links': written, 'reason': note}

    # -- batch ------------------------------------------------------------
    async def run(self, *, limit, person_ids=None, apply=True):
        ids = await asyncio.to_thread(self.people, limit, person_ids)
        gate, results, stop = asyncio.Semaphore(CONCURRENCY), [], asyncio.Event()

        async def one(person_id):
            async with gate:
                if stop.is_set():
                    return
                try:
                    results.append(await self.link(person_id, apply=apply))
                except BudgetUnavailable as exc:
                    stop.set()
                    results.append({'person': person_id, 'status': 'budget_wait', 'error': str(exc)})
                except Exception as exc:  # one card never blocks the others
                    logger.warning('event links for %s failed: %s', person_id, exc)
                    results.append({'person': person_id, 'status': 'error', 'error': str(exc)})
                    if apply:  # queue it behind the others for a later retry
                        try:
                            await asyncio.to_thread(self.mark, person_id, 'open', f'retry: {exc}'[:500])
                        except Exception as mark_exc:
                            logger.warning('event links retry mark for %s failed: %s', person_id, mark_exc)

        await asyncio.gather(*(one(i) for i in ids))
        return results


KINDS = HISTORY_RELATION_KINDS


def default_linker(store, config):
    """Production wiring: registry generation, shared narrow writer, enrichment RPC."""
    import os
    os.environ.setdefault('COMMULINGO_SUGGESTED_BY', CHANGED_BY)
    from llm.call_registry import generate_detailed, resolve
    from llm.gateway import estimate_cost_usd
    from commulingo.people import _HISTORY_RELATION_KINDS, normalize_commulingo_write, _run_edit
    from scripts.commulingo_gap_event_links import acceptable, next_sort_order
    from . import service
    assert tuple(_HISTORY_RELATION_KINDS) == KINDS, 'relation kinds drifted from the writer'

    async def generate(system, text, feature):
        profile = resolve(feature)
        result = await asyncio.to_thread(generate_detailed, feature, text, system=system, profile=profile)
        semantics = 'anthropic' if profile.provider in {'claude', 'deepseek_anthropic'} else 'openai'
        cost = estimate_cost_usd(profile.model, token_semantics=semantics, **result.usage) or 0
        if result.error or not result.text or result.truncated:
            # Transient: the person stays eligible for the next tick instead of being parked.
            raise RuntimeError(f'{feature}: {result.error_kind or "truncated"}: {result.error}')
        return result.text, cost

    def write(person_id, entry):
        patch = {'personId': person_id, 'sortOrder': next_sort_order(entry['event_id']),
                 'relationKind': entry['kind'],
                 'relation': {'ko': entry['relation_ko'].strip(), 'en': entry['relation_en'].strip()},
                 'note': {'ko': entry['note_ko'].strip(), 'en': entry['note_en'].strip()}}
        citations = [f"인물 카드 {person_id}: {entry['basis'][:300]}"]
        patch, citations, _confidence, _repairs = normalize_commulingo_write(
            'history_event_person', entry['event_id'], patch, citations, None)
        result = _run_edit('history_event_person', 'create', entry['event_id'], patch, citations, None)
        if result.startswith('Error:'):
            raise RuntimeError(result)

    def mark(person_id, status, reason):
        revision = service.call({'command': 'read', 'target': 'person', 'id': person_id})['revision']
        service.call({'command': 'enrichment', 'target': 'person', 'id': person_id, 'topic': 'events',
                      'expectedRevision': revision,
                      'status': status, 'reason': reason, 'sources': [f'person-card:{person_id}'],
                      'changedBy': CHANGED_BY, 'idempotencyKey': f'event-links:{person_id}:{uuid.uuid4().hex}'})

    return EventLinker(store, cap=config['daily_cap_usd'], review_fraction=config['review_fraction'],
                       generate=generate, write=write, mark=mark, acceptable=acceptable)
