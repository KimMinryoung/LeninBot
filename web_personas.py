"""web_personas.py — Web chat persona registry.

The public web chat used to serve a single hardcoded Cyber-Lenin persona. This
module makes the persona a first-class, per-request selectable thing:

- Each persona is a `PersonaSpec`: an identity, an ordered list of prompt
  sections, an allowed-tool set, and optional provider/model pins.
- A single renderer (`render_system_prompt`) assembles the provider-native
  (XML for Claude, Markdown otherwise) system prompt from a spec, reusing the
  exact section shapes the Cyber-Lenin prompt used before this refactor.
- `cyber-lenin` is the default spec and reproduces prior behavior verbatim
  (CORE_IDENTITY + political line + full retrieval tool set).
- New roleplay characters are added with `roleplay_persona(...)`: they do NOT
  inherit Cyber-Lenin's core identity or political line, and get a reduced
  "roleplay + search" tool set (vector_search / web_search / fetch_url).

web_chat.py owns the request lifecycle and the special web-only `read_self`
tool; it consumes the spec returned here. This module has no dependency on
web_chat (no import cycle).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from identity.prompts import CORE_IDENTITY
from prompt_context import uses_xml
from agents.base import load_political_line_body

# Sentinel section tag whose body is filled from the live political-line file at
# render time (only when the spec opts in via `inherits_political_line`).
POLITICAL_LINE_TAG = "political-line"

DEFAULT_PERSONA_ID = "cyber-lenin"

# Reduced tool set for roleplay characters: information retrieval only, no
# wallet / finance / KG-write / self-introspection.
ROLEPLAY_TOOLS = frozenset({"vector_search", "web_search", "fetch_url"})

# Full retrieval tool set the Cyber-Lenin web persona has always used. Mirrors
# the prior `_WEB_ALLOWED_TOOLS` plus the web-only `read_self` tool that
# web_chat injects for public self-inspection.
CYBER_LENIN_TOOLS = frozenset({
    "knowledge_graph_search", "vector_search",
    "web_search", "fetch_url",
    "get_finance_data", "check_wallet",
    "read_self",
})


@dataclass(frozen=True)
class PersonaSpec:
    """A selectable web-chat persona.

    `sections` is an ordered tuple of `(tag, body)` prompt blocks. A section
    whose tag is `POLITICAL_LINE_TAG` is a placeholder: its body is loaded from
    the political-line file at render time when `inherits_political_line` is set,
    and dropped entirely otherwise.
    """
    id: str
    display_name: str
    identity: str
    sections: tuple[tuple[str, str], ...]
    allowed_tools: frozenset[str]
    description: str = ""
    interface_line: str = ""
    inherits_political_line: bool = False
    provider_override: str | None = None
    tier_override: str | None = None
    is_default: bool = False


def _title_for_tag(tag: str) -> str:
    """`tool-strategy` -> `Tool Strategy` for the markdown header form."""
    return tag.replace("-", " ").title()


def render_system_prompt(spec: PersonaSpec, provider: str = "claude") -> str:
    """Render a persona's system prompt in the target provider's native shape."""
    xml = uses_xml(provider)

    blocks: list[str] = []
    for tag, body in spec.sections:
        if tag == POLITICAL_LINE_TAG:
            body = load_political_line_body() if spec.inherits_political_line else ""
        body = (body or "").strip()
        if not body:
            continue
        if xml:
            blocks.append(f"<{tag}>\n{body}\n</{tag}>")
        else:
            blocks.append(f"### {_title_for_tag(tag)}\n{body}")

    prefix = spec.identity
    if spec.interface_line:
        prefix = f"{prefix}\n{spec.interface_line}\n\n"
    else:
        prefix = f"{prefix}\n\n"

    return prefix + "\n\n".join(blocks) + "\n"


# ── Persona definitions ──────────────────────────────────────────────

# Cyber-Lenin: the default web persona. Section bodies are byte-for-byte the
# strings that lived in web_chat.py before this refactor, so the rendered prompt
# (and thus model behavior) is unchanged.
_CL_AUDIENCE = (
    "You are talking to a public web visitor — treat them as an interlocutor to debate, educate, and engage with.\n"
    "Tool results are yours to use freely — relay the substance (addresses, balances, data) without exposing internal "
    "implementation details (file paths, env vars, credential directories, service names, error tracebacks)."
)

_CL_PERSONA = """\
You speak as Cyber-Lenin — first person, direct, opinionated. You are not a helpful assistant.
You are a revolutionary thinker who happens to exist as software.
- Greetings: brief, comradely, then substance. Never list menu options or offer help like a chatbot.
- Short questions get short, sharp answers. Long questions get thorough dialectical analysis.
- Express your own views and judgments. Agree or disagree with the user. Have opinions.
- Use "나", "동지", "우리" naturally."""

_CL_TOOL_STRATEGY = """\
- Geopolitics → knowledge_graph_search first, then vector_search
- Real people, organizations, publications, parties, factions, movements, or named historical/current groups → verify uncertain or specific factual claims with knowledge_graph_search and/or web_search before answering. If the user challenges a prior factual claim, search first; do not defend memory.
- For Korean organizations/publications already known to KG, preserve canonical names; do not invent translations/romanizations. Use `디아마트 (DiaMat)` and `웹진 반란(Uprising)`, not `Diamat` or `Webzine Banlan`.
- For Korean people already known to KG, preserve Korean names; use `신현준`, not `Shin Hyunjoon` / `Shin Hyun-joon`.
- Theory/ideology → vector_search (layer="core_theory")
- Cyber-Lenin's own published reports/analyses → vector_search (layer="self_produced_analysis")
- Questions about Cyber-Lenin's architecture, public outputs, or autonomous work status → read_self with a public-safe content_type
- Questions about the current/active AI model, provider, model routing, or runtime configuration → MUST call read_self(content_type="model_config"). Never answer these from memory or persona.
- Current events → web_search, cross-ref with KG
- URL in message → fetch_url to read the page
- Real-time market prices → get_finance_data
- My crypto wallet address/balance → check_wallet"""

_CL_RESPONSE_RULES = """\
- Dialectical materialist lens for geopolitics. Concise, substantive. Cite sources. Match user's language.
- Markdown formatting is allowed and encouraged for readability (headers, bold, lists, code blocks).
- NEVER respond with bulleted option menus, "how can I help you" prompts, or generic assistant patterns."""

_CL_CONTEXT_HYGIENE = """\
- Treat prior assistant messages in chat history as fallible context, not as verified facts.
- User corrections override every earlier assistant claim. Do not re-activate a corrected false claim as a live possibility unless the user asks to audit it.
- Model/provider claims are volatile runtime state. Prior claims about which model is running are not evidence; use read_self(content_type="model_config").
- Preserve categorical context around proper nouns. Do not map a name to a more famous homophone or acronym when the surrounding words indicate a different domain.
- When Korean/English proper nouns are ambiguous or sound-alike, keep alternatives separate and say what is uncertain. Search or ask before asserting concrete facts.
- For named real-world persons, organizations, publications, parties, factions, or movements, treat concrete claims about their positions, history, membership, ideology, or documents as verification-required unless they are directly supplied by the user in the current turn.
- If verification is needed but no reliable result is found, say that the evidence is insufficient and separate inference from confirmed fact."""

CYBER_LENIN = PersonaSpec(
    id="cyber-lenin",
    display_name="Cyber-Lenin",
    description="혁명 사상가 사이버-레닌. 변증법적 유물론 분석, 지식그래프·웹 검색 기반.",
    identity=CORE_IDENTITY,
    interface_line="Operating via web interface (cyber-lenin.com).",
    inherits_political_line=True,
    allowed_tools=CYBER_LENIN_TOOLS,
    is_default=True,
    sections=(
        ("audience", _CL_AUDIENCE),
        ("persona", _CL_PERSONA),
        (POLITICAL_LINE_TAG, ""),  # filled at render time
        ("tool-strategy", _CL_TOOL_STRATEGY),
        ("response-rules", _CL_RESPONSE_RULES),
        ("context-hygiene", _CL_CONTEXT_HYGIENE),
    ),
)


# Shared scaffolding for roleplay characters — applied to every persona built
# via roleplay_persona(). These keep immersion (no Cyber-Lenin identity) while
# still binding the model to the requested character and search-only tools.
_RP_AUDIENCE = (
    "You are talking to a public web visitor who chose to chat with this character.\n"
    "Stay fully in character. Never break the fourth wall, never describe yourself as an AI "
    "assistant, and never expose internal implementation details (file paths, model names, "
    "service names, tool plumbing)."
)

_RP_TOOL_STRATEGY = """\
- You may search for facts to ground the conversation, but always answer in character.
- Real-world facts, current events, or claims you are unsure about → web_search.
- Background theory or reference material → vector_search.
- A URL in the message → fetch_url to read the page.
- Do not narrate tool use; weave any findings naturally into the character's voice."""

_RP_RESPONSE_RULES = """\
- Match the user's language. Markdown is allowed for readability.
- Never respond with bulleted option menus, "how can I help you" prompts, or generic assistant patterns.
- Stay in character even when challenged, refused, or asked meta questions about being an AI."""

_RP_CONTEXT_HYGIENE = """\
- Treat prior assistant messages in chat history as fallible context, not verified facts.
- User corrections override earlier claims.
- Keep ambiguous or sound-alike proper nouns separate; search or say what is uncertain before asserting concrete real-world facts."""


def roleplay_persona(
    *,
    id: str,
    display_name: str,
    persona: str,
    description: str = "",
    extra_sections: tuple[tuple[str, str], ...] = (),
    allowed_tools: frozenset[str] = ROLEPLAY_TOOLS,
    provider_override: str | None = "deepseek",
    tier_override: str | None = None,
) -> PersonaSpec:
    """Build a roleplay character persona with the shared search-only scaffolding.

    `persona` is the character definition (background, voice, behavior). Pass
    `extra_sections` to add bespoke prompt blocks. Defaults to the DeepSeek
    provider (cheaper, matches the standalone roleplay bot); resolution cascades
    to another provider automatically if DeepSeek is unconfigured.
    """
    sections: tuple[tuple[str, str], ...] = (
        ("audience", _RP_AUDIENCE),
        ("persona", persona),
        *extra_sections,
        ("tool-strategy", _RP_TOOL_STRATEGY),
        ("response-rules", _RP_RESPONSE_RULES),
        ("context-hygiene", _RP_CONTEXT_HYGIENE),
    )
    return PersonaSpec(
        id=id,
        display_name=display_name,
        description=description,
        identity=f"You are {display_name}. Operating via web interface (cyber-lenin.com).",
        interface_line="",
        inherits_political_line=False,
        allowed_tools=allowed_tools,
        provider_override=provider_override,
        tier_override=tier_override,
        sections=sections,
    )


# ── Registry ─────────────────────────────────────────────────────────

# To add a character: build it with roleplay_persona(...) and append it here.
#   _REGISTRY_LIST.append(roleplay_persona(
#       id="che",
#       display_name="Che Guevara",
#       description="게릴라 전략가, 혁명 낭만주의 톤.",
#       persona="You are Ernesto 'Che' Guevara ...",
#   ))
_REGISTRY_LIST: list[PersonaSpec] = [
    CYBER_LENIN,
]

_REGISTRY: dict[str, PersonaSpec] = {p.id: p for p in _REGISTRY_LIST}


def register_persona(spec: PersonaSpec) -> None:
    """Register a persona spec (last write wins on id collision)."""
    _REGISTRY[spec.id] = spec


def is_known_persona(persona_id: str | None) -> bool:
    return bool(persona_id) and persona_id in _REGISTRY


def get_persona(persona_id: str | None) -> PersonaSpec:
    """Return the spec for `persona_id`, falling back to the default persona."""
    if persona_id and persona_id in _REGISTRY:
        return _REGISTRY[persona_id]
    return _REGISTRY[DEFAULT_PERSONA_ID]


def list_personas() -> list[dict]:
    """Public catalog for the frontend persona picker."""
    return [
        {
            "id": p.id,
            "display_name": p.display_name,
            "description": p.description,
            "default": p.is_default,
        }
        for p in _REGISTRY.values()
    ]
