"""Generalized Reflexion pass: diagnose → author-revise for analytical content.

Ports the writer subsystem's blind-eval-validated conventions
(writer/prompts.py, writer/stream.py run_diagnose_revise_pass) to factual /
analytical text — task reports and autonomous research drafts:

- a cheap editor model DIAGNOSES (numbered notes anchored by verbatim quotes,
  direction of repair, never rewritten text; exactly ``PASS`` when clean);
- the AUTHOR model revises, free to reject notes that misread the work.

This module is provider-agnostic: callers supply a chat fn matching the
``chat_with_tools`` keyword interface and an already-resolved model id. The
writer's own implementation stays untouched as the reference for prose.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

DIAGNOSIS_PASS_MARKER = "PASS"

_KIND_DESCRIPTIONS = {
    "task_report": "an analytical report a research agent produced for a delegated task",
    "research_draft": "a research document draft staged for public publication",
}

_PUBLICATION_CHECK = (
    "5. 공개 적합성 — tone appropriate for public readers; claims that would need explicit "
    "sourcing before going public; internal working notes, TODOs, or agent self-talk left in "
    "the text.\n"
)


def diagnosis_is_pass(notes: str | None) -> bool:
    """True when the diagnosis found nothing worth an author revision."""
    stripped = (notes or "").strip()
    head = stripped.splitlines()[0].strip() if stripped else ""
    return head.upper().startswith(DIAGNOSIS_PASS_MARKER)


def _diagnosis_system_prompt(content_kind: str) -> str:
    kind_desc = _KIND_DESCRIPTIONS.get(content_kind, _KIND_DESCRIPTIONS["task_report"])
    publication_check = _PUBLICATION_CHECK if content_kind == "research_draft" else ""
    categories = "사실성|논리|완결성|명료성" + ("|공개적합성" if publication_check else "")
    return (
        f"You are a rigorous independent editor reviewing {kind_desc}. You DIAGNOSE; you never "
        "rewrite. Your notes go back to the author, who does the revising and is free to reject "
        "notes that misread the work. Your value is a fresh, unsparing outside eye.\n\n"
        "# Check, in order of leverage\n"
        "1. 사실성 — claims stated as fact without support in the text; numbers, dates, or names "
        "that are internally inconsistent; overclaiming beyond the evidence actually presented.\n"
        "2. 논리 — internal contradictions; conclusions that do not follow from the evidence "
        "given; obvious counter-arguments the text itself raises but never addresses.\n"
        "3. 완결성 — does it actually answer the stated task/goal? promised questions left "
        "unaddressed; the main finding buried under secondary material.\n"
        "4. 명료성 — redundancy, vague hedging, structure that hides the conclusion, unexplained "
        "jargon.\n"
        f"{publication_check}\n"
        "# Method\n"
        "Judge only against the text and the provided context — you have no tools; never invent "
        "outside facts to contradict the author. Be concrete and severe: praise is useless, and a "
        "missed real weakness is your only failure mode. Never write replacement text — name the "
        "problem and the direction of repair.\n\n"
        "# Response format\n"
        "ONLY a numbered diagnosis, most damaging first, at most 8 items:\n"
        f'1. [{categories}] "short verbatim quote locating the spot" — problem, then direction '
        "of repair (no rewritten text).\n"
        "Then one line — 남길 것: what is working and must survive revision.\n"
        f"If nothing rises to a real note, reply exactly {DIAGNOSIS_PASS_MARKER}."
    )


async def diagnose(
    content: str,
    *,
    chat_fn,
    model: str,
    content_kind: str = "task_report",
    context: str = "",
    max_tokens: int = 2000,
    budget_usd: float = 0.10,
    **chat_kwargs,
) -> str | None:
    """Run the editor diagnosis over ``content``. Returns the numbered notes,
    the bare PASS marker, or None when the stage produced nothing usable.
    Callers should treat any exception or None as "skip the revision"."""
    text = (content or "").strip()
    if not text:
        return None
    parts = [
        "Diagnose the text below. "
        f"Reply with the numbered diagnosis, or exactly {DIAGNOSIS_PASS_MARKER}."
    ]
    if context.strip():
        parts.append(f"[Context — what the text was supposed to accomplish]\n{context.strip()}")
    parts.append(f"[Text under review]\n{text}")
    notes = await chat_fn(
        [{"role": "user", "content": "\n\n".join(parts)}],
        system_prompt=_diagnosis_system_prompt(content_kind),
        model=model,
        max_tokens=max_tokens,
        budget_usd=budget_usd,
        **chat_kwargs,
    )
    return (notes or "").strip() or None


def build_report_revision_prompt(report: str, notes: str, *, context: str = "") -> str:
    """User message asking the AUTHOR model to revise its own report guided by
    (but not subordinate to) the editor's diagnosis. Text-only turn by design:
    revision must not re-run side-effectful tools."""
    parts = [
        "An outside editor reviewed the report you just wrote (notes below). Revise it as the "
        "author.\n"
        "- Judge each note: apply what is right, REJECT what misreads the work — obedience is "
        "not revision. Do not mention the notes or your rejections in the report itself.\n"
        "- You have no tools in this turn: where a note challenges a fact you cannot re-verify, "
        "qualify the claim honestly instead of inventing support.\n"
        "- Keep the report's language, format, and scope. Do not add new findings.\n"
        "Reply with ONLY the complete revised report — no preamble, no commentary.",
        f"[Editor's notes]\n{notes.strip()}",
    ]
    if context.strip():
        parts.append(f"[Original task]\n{context.strip()}")
    parts.append(f"[Your report]\n{report.strip()}")
    return "\n\n".join(parts)
