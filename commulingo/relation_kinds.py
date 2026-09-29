"""How a person relates to a history event: the one definition every writer reads.

The event page groups people by these kinds (frontend
data/commulingo/event-presentation.js holds the labels). Revised 2026-09-29:
executor had been labelled "주도 · 집행" on the page and defined as "carried out
orders" in the linking prompts, so content used it for prime movers and the
prompts for enforcers; commanders were spread over four kinds; opponent had no
reference point; witness mixed eyewitnesses with historians writing decades
later, which is why historian was added.
"""

HISTORY_RELATION_KIND_DEFINITIONS = {
    'leader': (
        "Set the course of one side or of the whole event at the top: head of state or party, "
        "supreme commander, the movement's organiser or chief decision-maker."),
    'executor': (
        "Carried out the leadership's decisions or commanded one part of the event: front, army "
        "or operation commander, security organ, prosecutor or judge, official implementing the policy."),
    'participant': (
        "Acted inside the event without directing it or being charged with carrying it out: "
        "delegate, soldier, member, signatory, author of one of its documents. A dispute inside "
        "the same side is participation, not opposition."),
    'opponent': (
        "Fought, resisted or tried to stop the process or side the event's title names: the old "
        "regime against a revolution, the enemy side in a war the event frames from one side, the "
        "counter-movement against a policy. In a clash with no central side, each side's people "
        "are leader, executor or participant."),
    'target': (
        "The event's action was done to them: arrested, tried, purged, executed, deported, deposed "
        "or attacked. When the caption says what happened to the person, this wins over opponent."),
    'witness': (
        "At the time, saw, reported or recorded the event without acting in it: war correspondent, "
        "diarist, foreign observer, contemporary writer."),
    'historian': (
        "Studied or interpreted the event afterwards: historian, scholar, later analyst whose "
        "work on this event is named."),
}

HISTORY_RELATION_KINDS = tuple(HISTORY_RELATION_KIND_DEFINITIONS)


def definitions_text(indent='      '):
    """The definitions as prompt lines, one kind per line."""
    return '\n'.join(f'{indent}{kind}: {text}' for kind, text in HISTORY_RELATION_KIND_DEFINITIONS.items())
