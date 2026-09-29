"""How a person relates to a history event: the one definition every writer reads.

The event page groups people by these kinds (frontend
data/commulingo/event-presentation.js holds the labels). Revised 2026-09-29:
executor had been labelled "주도 · 집행" on the page and defined as "carried out
orders" in the linking prompts, so content used it for prime movers and the
prompts for enforcers; commanders were spread over four kinds; opponent had no
reference point (fixed the same day by commulingo_history_events.focus, migration
192: opponents are the camp against the focus; in events without one, only
those who tried to stop the event itself);
witness mixed eyewitnesses with historians writing decades later, which is why
historian was added.

Named sides (frontend migration 193, 2026-09-29): an event with no single focus
may name its camps (commulingo_history_events.sides). There each person also
carries the side they acted for, and opponent is not used: the opposing camp is
simply another side. Events whose subject is clear keep focus + opponent.
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
        "the same side is participation, not opposition, and so is an outside government or figure "
        "who only reacted (diplomacy, statements, sanctions) without taking the opposing side."),
    'opponent': (
        "Stood on the side against the event's focus (given with the event), whatever their role "
        "there: the old regime against a revolution, the enemy army in a war, the counter-movement "
        "against a policy, the authorities who crushed an uprising. When the event has no focus, "
        "the only opponents are people who tried to stop the event itself (the war, the treaty, "
        "the plan, the coup), such as anti-war campaigners; people inside one camp who fought "
        "another camp are leader, executor or participant. Leader, executor and participant "
        "describe people on the focus side, or on any side when there is no focus. Never use "
        "opponent on an event that names its sides: there the opposing camp is another side."),
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


SIDE_RULE = (
    "An event that names sides (camps) takes a side for every person: the id of the camp they "
    "acted for, whatever their kind. Use null only for a witness, a historian, or someone who took "
    "no side (a mediator, a neutral government that only reacted). The kind is still their role "
    "inside that camp (a camp's own head is its leader, its commanders executors); opponent is "
    "never used there.")
