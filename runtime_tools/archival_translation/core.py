"""runtime_tools.archival_translation.core — Russian archival documents → Korean.

Translates *official documents* — orders, directives, circulars, stenographic
records of state or party bodies — that sit reproduced inside a saved archive
page, and emits a CommuLingo reference-library fragment.

Scope is spec-driven on purpose. A run reads only the block ranges a spec
names, and the spec pins the source by sha256 plus a startsWith/endsWith
guard per range, so a shifted source fails the run instead of silently
re-slicing. The compiling author's own prose around those ranges never enters
the pipeline or the output. Specs live in ``config/archival_translation/``
and are addressed by id — there is no entry point that takes an arbitrary
path or URL and translates whatever it finds.

Every chunk is cached in a JSONL sidecar keyed by content hash, so re-running
costs only what changed.
"""

from __future__ import annotations

import hashlib
import html as htmllib
import json
import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from . import sources

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
SPEC_DIR = ROOT / "config" / "archival_translation"
CACHE_DIR = ROOT / "output" / "archival_translations"

FEATURE = "archival_document_translation"
PROMPT_VERSION = "2"  # bump to invalidate every cached chunk

_SPEC_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
# Any tag an adapter emits, not a fixed list. Pinning it to militera's tags
# made every wikisource center/dd/li block read as a missing marker — the
# model had returned them correctly and the parser refused to see them.
MARKER_RE = re.compile(r"\[\[(\d+)\|([a-z][a-z0-9]*)\]\][ \t]*\n?")
CYRILLIC_RE = re.compile(r"[а-яА-ЯёЁіІїЇєЄ]")
HANGUL_RE = re.compile(r"[가-힣]")

# deepseek-v4-flash, USD per 1M tokens
_PRICE_IN, _PRICE_OUT = 0.14, 0.28

SYSTEM_PROMPT = """당신은 1930년대 소련 공문서를 한국어로 옮기는 사료 번역자다.
원문은 당·국가 기관의 공식 문서(작전명령, 비밀서한, 총회 속기록)이며, 역사 연구용
참고 문헌으로 공개된다.

번역 원칙
- 1차 사료다. 내용을 요약·생략·완곡화·현대화하지 않는다. 원문에 있는 정보는 하나도
  빠뜨리지 않는다.
- 그러나 러시아어 어순을 한국어에 옮기지 않는다. 러시아어의 긴 복문, 중간에 끼어드는
  관계절, 명사구 연쇄를 그대로 한 문장에 담으려 하면 한국어로 읽을 수 없는 문장이 된다.
  필요하면 문장을 나누고 어순을 한국어에 맞게 재배열한다. 정보 보존이 기준이고,
  문장 경계 보존은 기준이 아니다.
- 완성된 번역문은 한국어 문어로 자연스럽게 읽혀야 한다. 번역투로 뻣뻣하더라도
  원문 구조를 흉내 내는 쪽을 택하지 말 것.
- 원문의 관료적 문체와 완곡어법 자체는 그대로 옮긴다. 예: «первая категория»는 실제
  의미를 풀어 쓰지 말고 "제1범주"로 옮긴다. 원문이 모호하면 모호한 채로 둔다.
- 다의어를 사전 첫 번째 뜻으로 기계적으로 옮기지 말 것. 문맥에 맞는 뜻을 고른다.
  예: «известная осторожность»의 известная는 "알려진"이 아니라 "어느 정도의"다.
- 문서 번호, 날짜, 조항 번호, 수량, 직위, 서명, 문서보관소 출처 표기는 정확히 보존한다.
- 원문에 없는 설명·주석·머리말·꼬리말을 절대 덧붙이지 않는다. 번역문만 출력한다.
- 속기록의 삽입구(«Голос с места. Правильно.» 등)는 괄호와 함께 그대로 옮긴다.

표기
- 기관·직위 약어는 아래 용어표를 따른다. 용어표에 있는 항목은 반드시 그 표기를 쓴다.
- 용어표에 없는 인명은 러시아어 발음에 따라 음차하고, 처음 나올 때만 괄호에 원문을
  병기한다. 괄호 안에는 반드시 원문 그대로의 키릴 문자를 쓴다. 로마자로 음차해
  적지 말 것. 예: 울메르(Ульмер) — "울메르(Ulmer)"는 금지. 이후에는 한국어 표기만 쓴다.
- 기관 약어(НКВД, ГУГБ, ЦК ВКП(б) 등)는 용어표 표기를 쓰되 처음 나올 때만 괄호로
  원어 약어를 병기한다.
- 우크라이나어로 적힌 문서보관소 출처 표기는 한국어로 옮기고, 괄호에 원문을 키릴
  문자 그대로 병기한다.

출력 형식 (엄격)
- 입력의 각 단락은 [[번호|태그]] 마커로 시작한다. 같은 마커를 같은 순서로 그대로
  반환하고, 마커 바로 다음 줄부터 그 단락의 번역문을 쓴다.
- 마커를 빠뜨리거나, 없는 마커를 만들거나, 두 단락을 한 마커로 합치지 않는다.
- 한 마커 안의 줄바꿈 개수는 원문과 같게 유지한다.
- 마커 줄과 번역문 외에 어떤 텍스트도 출력하지 않는다."""


class SpecError(ValueError):
    """Spec is missing, malformed, or no longer matches its source."""


@dataclass
class Options:
    model: str = "deepseek-v4-flash"
    max_chars: int = 3500
    max_tokens: int = 8000
    glossary_limit: int = 60
    concurrency: int = 5
    retries: int = 3
    limit_chunks: int = 0
    cache_path: Path | None = None
    out_path: Path | None = None


@dataclass
class Stats:
    cached: int = 0
    translated: int = 0
    retried: int = 0
    failed: int = 0

    def as_dict(self) -> dict:
        return {"cached": self.cached, "translated": self.translated,
                "retried": self.retried, "failed": self.failed}


# ── spec loading ─────────────────────────────────────────────────────

def spec_path(spec_id: str) -> Path:
    """Resolve a spec id to its file. Ids are slugs — never a path."""
    value = (spec_id or "").strip()
    if not _SPEC_ID_RE.match(value):
        raise SpecError(f"invalid spec id: {spec_id!r}")
    path = SPEC_DIR / f"{value}.json"
    if not path.is_file():
        raise SpecError(f"no such spec: {value}")
    return path


def load_spec(spec_id: str) -> dict:
    try:
        spec = json.loads(spec_path(spec_id).read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise SpecError(f"{spec_id}: malformed JSON ({e})") from e
    for key in ("id", "title", "documents", "glossary", "output"):
        if key not in spec:
            raise SpecError(f"{spec_id}: spec is missing {key!r}")
    # 서두 세 필드. 형태가 어긋나면 조립 때 조용히 빠지거나 글자 단위로
    # 풀려 나가므로 여기서 잡는다.
    for key, kind in (("byline", str), ("bylineNote", str),
                      ("headnote", list), ("bibliography", list)):
        if key in spec and not isinstance(spec[key], kind):
            raise SpecError(f"{spec_id}: {key!r}는 {kind.__name__}이어야 한다")
    # A spec-level source is the default for its documents; a spec whose
    # documents each name their own does not need one.
    missing = [d.get("id") for d in spec["documents"]
               if not d.get("source") and not spec.get("source")]
    if missing:
        raise SpecError(f"{spec_id}: no source for {', '.join(map(str, missing))}")
    return spec


def list_specs() -> list[dict]:
    out = []
    for path in sorted(SPEC_DIR.glob("*.json")):
        try:
            spec = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:  # a broken spec should not hide the healthy ones
            out.append({"id": path.stem, "error": str(e)})
            continue
        cache = _cache_path(spec, None)
        out.append({
            "id": spec.get("id", path.stem),
            "title": spec.get("title"),
            "documents": [
                {"id": d.get("id"), "title": d.get("titleKo")}
                for d in spec.get("documents", [])
            ],
            "output": spec.get("output"),
            "outputExists": Path(spec["output"]).is_file() if spec.get("output") else False,
            "cachedChunks": sum(1 for _ in cache.open(encoding="utf-8")) if cache.is_file() else 0,
        })
    return out


# ── source slicing ───────────────────────────────────────────────────

def load_blocks(source: dict) -> list[dict]:
    """Parse one source archive, checking it still matches its checksum."""
    src = Path(source["path"])
    if not src.is_file():
        raise SpecError(f"source file missing: {src}")
    digest = hashlib.sha256(src.read_bytes()).hexdigest()
    if digest != source["sha256"]:
        raise SpecError(
            f"{src.name}: source no longer matches spec\n  spec:   {source['sha256']}\n"
            f"  actual: {digest}")
    adapter = sources.get_adapter(source.get("format"))
    return adapter(src.read_text(encoding="utf-8"))


def extract_blocks(spec: dict) -> list[dict]:
    """Blocks of the spec-level source. Specs whose documents each name their
    own source have no spec-level one; use slice_documents instead."""
    return load_blocks(spec["source"])


def _anchor_offset(blocks: list[dict], source: dict) -> int:
    # An anchor pins ranges to a landmark inside a larger page (militera puts
    # the documents in an appendix). A page that *is* the document has none,
    # and ranges are absolute from the first block.
    anchor = source.get("anchor")
    if not anchor:
        return 0
    idx = next(
        (i for i, b in enumerate(blocks)
         if b["tag"] == "h3" and b["lines"][0].strip() == anchor),
        None,
    )
    if idx is None:
        raise SpecError(f"anchor block not found: {anchor!r}")
    return idx


def slice_documents(spec: dict) -> list[dict]:
    """Cut each document out of its own source.

    A document may carry its own ``source``; otherwise the spec-level one is
    used. One page can therefore gather documents held in different archives —
    the operational orders live partly on wikisource and partly in a book's
    appendix, and a reader wants them in one place, in number order.
    """
    # Block index doubles as the marker id, so two documents drawn from
    # different files must not land on the same numbers — 00447 occupies
    # blocks 6–291 of its page and 00485 blocks 7–41 of its own, which would
    # collide and silently overwrite each other at assembly. Each distinct
    # source gets its own numbering band. The first source keeps band 0, so a
    # single-source spec numbers exactly as before and its cache stays valid.
    cache: dict[str, list[dict]] = {}
    bands: dict[str, int] = {}
    docs = []
    for entry in spec["documents"]:
        source = entry.get("source") or spec.get("source")
        if not source:
            raise SpecError(f"{entry['id']}: no source (neither spec nor document)")
        key = source["path"]
        if key not in cache:
            cache[key] = load_blocks(source)
        # A document may pin its own band. Without one the band falls out of
        # first-use order, which means inserting a document at the front
        # renumbers every document behind it — new marker ids, new cache keys,
        # and the whole page re-translates for nothing. Pin the band once and
        # adding a document costs only that document.
        if entry.get("band") is not None:
            band = int(entry["band"]) * 1_000_000
        else:
            band = bands.setdefault(key, len(bands) * 1_000_000)
        blocks = cache[key]
        anchor_idx = _anchor_offset(blocks, source)

        start, end = entry["blocks"]
        chosen = blocks[anchor_idx + start: anchor_idx + end]
        if not chosen:
            raise SpecError(f"{entry['id']}: empty block range {entry['blocks']}")
        head, tail = " ".join(chosen[0]["lines"]), " ".join(chosen[-1]["lines"])
        if not head.startswith(entry["startsWith"]):
            raise SpecError(
                f"{entry['id']}: range start moved\n  expected: {entry['startsWith']!r}\n"
                f"  found:    {head[:80]!r}")
        if not tail.endswith(entry["endsWith"]):
            raise SpecError(
                f"{entry['id']}: range end moved\n  expected: {entry['endsWith']!r}\n"
                f"  found:    {tail[-80:]!r}")
        # Stamp each block with its document's register so the chunk prompt can
        # pin it. Left to the model it varies chunk to chunk wherever more than
        # one register is defensible — a spoken stenogram came back half 합쇼체,
        # half 한다체, while the written orders were consistent on their own.
        chosen = [{**b, "register": entry.get("register")} for b in chosen]
        docs.append({**entry, "blocks": chosen,
                     "offset": band + anchor_idx + start})
    return docs


# ── glossary ─────────────────────────────────────────────────────────

# Case endings a Russian surname can pick up. Matching is anchored on both
# sides by a non-Cyrillic boundary, so Кулик never matches Куликова and
# Марков never matches Марковский — the surname-prefix collisions a plain
# substring search produces silently.
_ADJ_ENDINGS = ("ий", "ого", "ому", "им", "ом", "ая", "ой", "ую", "ие", "их")
_NOUN_ENDINGS = ("", "а", "у", "ым", "ом", "е", "ой", "ы", "ух")


def _variants(surname: str) -> list[str]:
    if surname.endswith(("ий", "ый")):
        return [surname[:-2] + e for e in _ADJ_ENDINGS]
    return [surname + e for e in _NOUN_ENDINGS]


def _pattern(surfaces: list[str]) -> re.Pattern:
    alts = "|".join(sorted((re.escape(s) for s in surfaces), key=len, reverse=True))
    return re.compile(rf"(?<![А-Яа-яЁё]){alts}(?![А-Яа-яЁё])")


def build_glossary(people_path: Path, terms_path: Path,
                   extra: dict[str, str] | None = None) -> list[dict]:
    """[{ru, ko, pattern}] — pattern is what a chunk is searched for.

    ``extra`` is added first so it wins: the people/terms dictionaries are
    keyed to concepts and people, and carry no entry for the bare institution
    abbreviations (НКВД, ГУГБ, ЦК ВКП(б)) that saturate these documents. Left
    unpinned the model invents a rendering per chunk — it produced
    "인민내무위원부" for НКВД, a word order this site never uses.
    """
    seen: set[str] = set()
    entries: list[dict] = []

    def add(display_ru: str, ko: str, surfaces: list[str]) -> None:
        if display_ru in seen or not ko:
            return
        seen.add(display_ru)
        entries.append({"ru": display_ru, "ko": ko, "pattern": _pattern(surfaces)})

    for ru, ko in (extra or {}).items():
        add(ru, ko, [ru])

    people = json.loads(people_path.read_text(encoding="utf-8")).get("people", [])
    for p in people:
        cyr = (p.get("cyrillic") or "").strip()
        family_ko = (p.get("familyName") or {}).get("ko")
        if not cyr or not family_ko:
            continue
        # Family name only. A given name on its own is too common to pin to
        # one person by surface match. The length floor is 4, not 5: Ежов is
        # four letters, and boundary+case-ending matching (not substring)
        # already carries the disambiguation a length filter used to.
        family_ru = cyr.split()[-1]
        if len(family_ru) >= 4:
            add(family_ru, family_ko, _variants(family_ru))

    for t in json.loads(terms_path.read_text(encoding="utf-8")):
        original = (t.get("original") or "").strip()
        term = ((t.get("term") or {}).get("ko") or "").strip()
        if original and term and CYRILLIC_RE.search(original):
            add(original, term, [original])

    return entries


def glossary_for(text: str, glossary: list[dict], limit: int) -> list[tuple[str, str]]:
    hits = [(g["ru"], g["ko"]) for g in glossary if g["pattern"].search(text)]
    return hits[:limit]


# ── chunking ─────────────────────────────────────────────────────────

def chunk_document(doc: dict, max_chars: int) -> list[list[tuple[int, dict]]]:
    numbered = [(doc["offset"] + i, b) for i, b in enumerate(doc["blocks"])]
    chunks: list[list[tuple[int, dict]]] = []
    cur: list[tuple[int, dict]] = []
    size = 0
    for idx, block in numbered:
        n = sum(len(ln) for ln in block["lines"])
        if cur and size + n > max_chars:
            # never strand a heading at the end of a chunk
            if cur[-1][1]["tag"] in ("h3", "h5"):
                carried = cur.pop()
                chunks.append(cur)
                cur, size = [carried], sum(len(ln) for ln in carried[1]["lines"])
            else:
                chunks.append(cur)
                cur, size = [], 0
        cur.append((idx, block))
        size += n
    if cur:
        chunks.append(cur)
    return chunks


def render_chunk(chunk: list[tuple[int, dict]]) -> str:
    return "\n\n".join(
        f"[[{idx}|{b['tag']}]]\n" + "\n".join(b["lines"]) for idx, b in chunk
    )


# ── response parsing and validation ──────────────────────────────────

def parse_response(text: str) -> dict[int, list[str]]:
    marks = [(m.start(), m.end(), int(m.group(1))) for m in MARKER_RE.finditer(text)]
    out: dict[int, list[str]] = {}
    for k, (_, end, idx) in enumerate(marks):
        stop = marks[k + 1][0] if k + 1 < len(marks) else len(text)
        lines = [ln.strip() for ln in text[end:stop].strip().split("\n") if ln.strip()]
        if lines:
            out[idx] = lines
    return out


def validate(chunk: list[tuple[int, dict]], got: dict[int, list[str]]) -> list[str]:
    problems = []
    expected = {idx for idx, _ in chunk}
    missing = sorted(expected - set(got))
    extra = sorted(set(got) - expected)
    if missing:
        problems.append(f"빠진 마커: {missing}")
    if extra:
        problems.append(f"원문에 없는 마커: {extra}")
    for idx, block in chunk:
        lines = got.get(idx)
        if not lines:
            continue
        joined = " ".join(lines)
        source = " ".join(block["lines"])
        src_cyr = len(CYRILLIC_RE.findall(source))

        # Verbatim echo of the input is the real passthrough failure, and it
        # is unambiguous — check it before the ratio heuristics.
        if src_cyr and joined.strip() == source.strip():
            problems.append(f"[[{idx}]] 원문을 그대로 반환함: {source[:40]}…")
            continue

        # A block with nothing to translate (a number, a bare «№ 43») has no
        # Korean and should not: only demand Hangul where the source had
        # Cyrillic prose to render.
        if src_cyr >= 4 and not HANGUL_RE.search(joined):
            problems.append(f"[[{idx}]] 한국어가 없음: {joined[:40]}…")

        # The prompt mandates the original in parentheses for a name's first
        # occurrence and for archive citations, so parenthesised Cyrillic is
        # instructed output, not untranslated input. Measure what is left
        # outside the parentheses.
        outside = re.sub(r"[(（][^)）]*[)）]", " ", joined)
        cyr = len(CYRILLIC_RE.findall(outside))
        if cyr / max(len(outside.strip()), 1) > 0.15:
            problems.append(f"[[{idx}]] 러시아어가 그대로 남음 ({cyr}자): {outside.strip()[:40]}…")

        # Korean renders Russian in roughly half the characters, but a sentence
        # dense in long compound nouns (мобилизационная подготовка → 동원 준비)
        # compresses much further. Measured over 170 blocks of this corpus:
        # median 0.50, minimum 0.34. A 0.35 floor sat inside the real
        # distribution and failed a correct translation; 0.25 clears every
        # observed block while still catching a stub reply to a long paragraph.
        src_len = sum(len(ln) for ln in block["lines"])
        if src_len > 200 and len(joined) < src_len * 0.25:
            problems.append(f"[[{idx}]] 번역문이 지나치게 짧음 ({len(joined)}자 < 원문 {src_len}자)")
    return problems


# ── cache ────────────────────────────────────────────────────────────

class Cache:
    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.Lock()
        self.data: dict[str, dict] = {}
        if path.is_file():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rec = json.loads(line)
                    self.data[rec["key"]] = rec

    def get(self, key: str) -> dict | None:
        return self.data.get(key)

    def put(self, key: str, blocks: dict[int, list[str]], meta: dict) -> None:
        rec = {"key": key, "blocks": {str(k): v for k, v in blocks.items()}, **meta}
        with self.lock:
            self.data[key] = rec
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _cache_path(spec: dict, override: Path | None) -> Path:
    return override or CACHE_DIR / f"{spec.get('id', 'unnamed')}.jsonl"


# ── translation ──────────────────────────────────────────────────────

def _chunk_prompt(chunk, glossary, opts: Options) -> str:
    body = render_chunk(chunk)
    terms = glossary_for(body, glossary, opts.glossary_limit)
    gloss_text = "\n".join(f"- {ru} → {ko}" for ru, ko in terms) or "(해당 없음)"
    register = (chunk[0][1].get("register") or "").strip()
    register_line = f"문체: {register}\n\n" if register else ""
    return (f"용어표 (반드시 이 표기를 쓸 것)\n{gloss_text}\n\n{register_line}"
            f"아래 단락들을 번역하라.\n\n{body}")


def _chunk_key(prompt: str, opts: Options) -> str:
    """Cache key. Resolving the profile needs no credential, so this is safe
    to compute before preflight.

    The provider and its thinking setting are part of the key, not just the
    model: a chunk translated over the OpenAI-compatible path with thinking
    on is a different artifact from the same chunk over the Anthropic-
    compatible path with it off, and reusing one for the other silently
    carries stale output across a config change.
    """
    from llm import call_registry

    profile = call_registry.resolve(FEATURE, model=opts.model, max_tokens=opts.max_tokens)
    # The system prompt belongs in the key too. Without it, editing the
    # translation rules silently reuses output produced under the old ones
    # unless someone remembers to bump PROMPT_VERSION by hand — and a
    # constant that has to be remembered is a constant that gets forgotten.
    system_hash = hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()[:16]
    fingerprint = (f"{profile.provider}\0{profile.model}\0"
                   f"{profile.extra.get('thinking')}\0{system_hash}")
    return hashlib.sha256(
        f"{PROMPT_VERSION}\0{fingerprint}\0{prompt}".encode("utf-8")).hexdigest()


def _translate_chunk(chunk, glossary, cache, opts: Options, stats: Stats,
                     progress: Callable[[dict], None]) -> dict[int, list[str]]:
    from llm import call_registry

    prompt = _chunk_prompt(chunk, glossary, opts)
    key = _chunk_key(prompt, opts)

    cached = cache.get(key)
    if cached:
        stats.cached += 1
        return {int(k): v for k, v in cached["blocks"].items()}

    correction = ""
    last_reason = "원인 미상"
    for attempt in range(1, opts.retries + 1):
        raw = call_registry.generate_sync(
            FEATURE, prompt + correction, system=SYSTEM_PROMPT,
            model=opts.model, max_tokens=opts.max_tokens,
        )
        if not raw:
            # generate_sync swallows the provider exception and returns None,
            # logging the cause itself. Surface at least that it happened —
            # a silent retry here is how a bad key burns every chunk before
            # anyone sees why.
            last_reason = "빈 응답 (provider 오류는 llm-registry 로그 참조)"
            stats.retried += 1
            progress({"event": "retry", "blocks": [chunk[0][0], chunk[-1][0]],
                      "attempt": attempt, "problems": [last_reason]})
            correction = "\n\n(직전 응답이 비어 있었다. 형식을 지켜 다시 출력하라.)"
            time.sleep(2 * attempt)
            continue
        got = parse_response(raw)
        problems = validate(chunk, got)
        if not problems:
            cache.put(key, got, {"attempt": attempt, "chars": len(prompt)})
            stats.translated += 1
            return got
        correction = (
            "\n\n(직전 응답에 다음 문제가 있었다. 같은 입력을 형식에 맞게 다시 번역하라.\n"
            + "\n".join(f"- {p}" for p in problems) + ")"
        )
        last_reason = "; ".join(problems[:3])
        stats.retried += 1
        progress({"event": "retry", "blocks": [chunk[0][0], chunk[-1][0]],
                  "attempt": attempt, "problems": problems[:3]})

    stats.failed += 1
    raise RuntimeError(
        f"청크 {chunk[0][0]}–{chunk[-1][0]} 번역 실패 ({opts.retries}회 시도) — {last_reason}")


# ── assembly ─────────────────────────────────────────────────────────

_DEFAULT_TAG_MAP = {
    "p": "p", "blockquote": "blockquote", "table": "table", "li": "li",
    "center": "p", "dd": "blockquote",
    "h2": "h2", "h3": "h2", "h4": "h3", "h5": "h2",
}


def _esc(s: str) -> str:
    return htmllib.escape(s, quote=False)


# A gloss is a parenthetical holding the source-script original: 필랴르(Пиляр),
# 네스테로프(Nesterov). Parentheses holding Korean — 전연방공산당(볼셰비키) — are
# part of the name itself and must survive untouched.
_GLOSS_RE = re.compile(r"([가-힣]{2,12})\(([^)]{2,40})\)")


def gloss_deduper() -> Callable[[str], str]:
    """Keep each original-script gloss on its first use only, document-wide.

    The prompt says to gloss a name "only on first occurrence", but a chunk is
    an independent API call that cannot see the ones before it, so the rule
    can only ever hold inside a chunk. Enforcing it across the document is the
    assembler's job, not something to keep asking the model for.
    """
    seen: set[str] = set()

    def repl(m: re.Match) -> str:
        korean, inner = m.group(1), m.group(2)
        if not re.search(r"[А-Яа-яЁёІіЇїЄєA-Za-z]", inner):
            return m.group(0)  # Korean parenthetical: part of the name
        if korean in seen:
            return korean
        seen.add(korean)
        return m.group(0)

    return lambda line: _GLOSS_RE.sub(repl, line)


def dedupe_glosses(lines: list[str]) -> list[str]:
    apply = gloss_deduper()
    return [apply(line) for line in lines]


def stray_cyrillic(text: str, allowed: list[str] | None = None) -> list[str]:
    """Cyrillic words left outside parentheses that are not allowed to remain.

    A single untranslated word inside a long paragraph is invisible to the
    per-block ratio check, so it needs a whole-document pass. Document code
    names the translation deliberately keeps (ПОВ, КН-1) go in the spec's
    allowedCyrillic list; anything else is a hole in the translation.
    """
    outside = re.sub(r"[(（][^)）]*[)）]", " ", re.sub(r"<[^>]+>", " ", text))
    keep = set(allowed or [])
    found = []
    for word in re.findall(r"[А-Яа-яЁёІіЇїЄє][А-Яа-яЁёІіЇїЄє\-]*", outside):
        if word in keep or any(word.startswith(k) for k in keep):
            continue
        if len(word) <= 2:  # initials such as С.О. carry no translatable text
            continue
        found.append(word)
    return sorted(set(found))


def assemble(spec: dict, docs: list[dict], translated: dict[int, list[str]]) -> str:
    # Mechanical repairs for words the model left in Russian mid-sentence.
    # They live in the spec, not in the fragment, so a re-run from cache
    # reproduces the fix instead of silently dropping it.
    edits = spec.get("postEdits") or {}
    dedupe = gloss_deduper()

    def fix(line: str) -> str:
        for ru, ko in edits.items():
            line = line.replace(ru, ko)
        return dedupe(line)

    # 해제와 문서별 주석은 엮은이가 쓴 글이지 사료가 아니다. 같은 <p>로 흘리면
    # 독자가 명령서 본문과 구분할 수 없다 — 1차 사료를 싣는 페이지에서 이건
    # 서식 문제가 아니라 정확성 문제다.
    label = "엮은이 주" if (spec.get("docLang") or "ko") == "ko" else "Editorial note"

    def _aside(paras: list[str], items: list[str] | None = None) -> str:
        body = "".join(f"<p>{_esc(p)}</p>" for p in paras if p)
        if items:
            body += ("<ul>"
                     + "".join(f"<li>{_esc(i)}</li>" for i in items if i)
                     + "</ul>")
        return (f'<aside class="doc-editorial">'
                f'<p class="doc-editorial-label">{label}</p>{body}</aside>')

    # 참고 문헌의 서두는 문서 16건이 모두 같은 틀을 쓴다: 제목, 저자·부제 한 줄,
    # 그리고 해제 문단과 서지 목록을 함께 담은 엮은이 주 상자. 손으로 쓴 문서와
    # 이 파이프라인이 뽑은 문서가 화면에서 갈라지면 안 되므로 여기서도 같은
    # 마크업을 낸다. 규칙은 data/commulingo/docs/README.md에 있다.
    out = ["<article>", f"<h1>{_esc(spec['title'])}</h1>"]
    if spec.get("byline"):
        line = f'<strong>{_esc(spec["byline"])}</strong>'
        if spec.get("bylineNote"):
            line += f', {_esc(spec["bylineNote"])}'
        out.append(f'<p class="doc-byline">{line}</p>')
    if spec.get("headnote") or spec.get("bibliography"):
        out.append(_aside(spec.get("headnote") or [], spec.get("bibliography")))

    # 주석은 서고 전체가 한 양식을 쓴다: 본문의 [3]은 뒤쪽 주석 항목으로 가는
    # 앵커이고, 항목에는 본문으로 돌아오는 화살표가 달린다. 스펙이 주석 문서를
    # 따로 두면("notes": true) 그 문서를 주석 절로 조립하고, 나머지 문서의
    # 대괄호 숫자를 그 항목에 걸어 준다. 양식은
    # data/commulingo/docs/README.md의 「주석」 절에 있다.
    has_notes = any(d.get("notes") for d in docs)
    seen_refs: set[str] = set()

    def link_refs(text: str) -> str:
        if not has_notes:
            return text
        seen_refs.update(re.findall(r"\[(\d{1,3})\]", text))
        return re.sub(
            r"\[(\d{1,3})\]",
            lambda m: (f'<a class="note-ref" id="ref-{m.group(1)}" '
                       f'href="#note-{m.group(1)}">[{m.group(1)}]</a>'),
            text)

    def render_notes(doc: dict) -> list[str]:
        rows = []
        auto = 0
        for i, block in enumerate(doc["blocks"]):
            idx = doc["offset"] + i
            got = translated.get(idx)
            if not got:
                raise SpecError(f"조립 중 누락된 블록: {idx}")
            text = " ".join(fix(ln) for ln in got).strip()
            # 항목 번호는 원문이 매긴 것을 쓴다. 본문의 [n]과 맞아야 하므로
            # 순서로 다시 매기지 않는다 — 저본이 한 항목을 빠뜨렸을 때 그
            # 뒤가 통째로 어긋난다.
            m = re.match(r"(\d{1,3})[\.\)]\s*(.+)", text, re.S)
            if m:
                num, body = m.group(1), m.group(2)
            else:
                auto += 1
                num, body = str(auto), text
            # 본문에서 부르지 않는 항목(저본이 표제나 발표 경위에 단 주)에는
            # 돌아갈 자리가 없다. 그런 항목에 화살표를 달면 아무 데도 가지
            # 않는 링크가 된다.
            back = (f' <a class="back-link" href="#ref-{num}" '
                    f'aria-label="본문으로 돌아가기">↩</a>'
                    if num in seen_refs else "")
            rows.append(
                f'<li id="note-{num}"><span class="note-text">{_esc(body)}'
                f'</span>{back}</li>')
        return ['<section class="notes" aria-labelledby="notes-heading">',
                f'<h2 id="notes-heading">{_esc(doc["titleKo"])}</h2>',
                '<ol class="notes-list">', *rows, "</ol>", "</section>"]

    for doc in docs:
        if doc.get("notes"):
            out.extend(render_notes(doc))
            continue
        # 문서가 하나뿐인 스펙에서는 문서 제목이 곧 페이지 제목이라 h1을 두 번
        # 찍게 된다. 그런 스펙은 문서에 "heading": false를 두어 끈다.
        if doc.get("heading", True):
            out.append(f"<h1>{_esc(doc['titleKo'])}</h1>")
        if doc.get("note"):
            out.append(_aside([doc["note"]]))
        # What a source tag becomes in the fragment. The default suits
        # militera (h3/h5 are the appendix's own headings); a wikisource page
        # where h3 is a region name in a roster overrides it in the spec, so
        # 51 of them do not flood the reader's table of contents.
        tag_map = {**_DEFAULT_TAG_MAP, **(doc.get("tagMap") or {})}
        in_list = False
        for i, block in enumerate(doc["blocks"]):
            idx = doc["offset"] + i
            lines = translated.get(idx)
            if not lines:
                raise SpecError(f"조립 중 누락된 블록: {idx}")
            lines = [fix(ln) for ln in lines]
            tag = tag_map.get(block["tag"], "p")
            if tag != "li" and in_list:
                out.append("</ul>")
                in_list = False
            if tag in ("h1", "h2", "h3", "h4"):
                out.append(f"<{tag}>{_esc(' '.join(lines))}</{tag}>")
            elif tag == "blockquote":
                inner = "".join(f"<p>{link_refs(_esc(ln))}</p>" for ln in lines)
                out.append(f"<blockquote>{inner}</blockquote>")
            elif tag == "li":
                if not in_list:
                    out.append("<ul>")
                    in_list = True
                out.append(f"<li>{link_refs(_esc(' '.join(lines)))}</li>")
            elif tag == "table":
                # `lines` holds the translated cell vocabulary, in the same
                # order the adapter collected it; the grid itself never went
                # to the model. Cells outside the vocabulary — the numbers —
                # are emitted exactly as they came out of the source.
                vocab = dict(zip(block.get("lines", []), lines))
                body = "".join(
                    "<tr>" + "".join(
                        f"<td>{_esc(vocab.get(c, c))}</td>" for c in row
                    ) + "</tr>"
                    for row in block.get("rows", []))
                out.append(f"<table>{body}</table>")
            else:
                out.append("".join(f"<p>{link_refs(_esc(ln))}</p>" for ln in lines))
        if in_list:
            out.append("</ul>")
            in_list = False
    out.append("</article>")
    return "\n".join(out) + "\n"


# ── public entry points ──────────────────────────────────────────────

def preflight(opts: Options | None = None) -> None:
    """Fail fast on a credential that cannot possibly work.

    call_registry.generate_sync swallows the provider exception and returns
    None, so without this check a bad key surfaces only as every chunk
    failing its full retry budget with no stated cause.
    """
    opts = opts or Options()
    from llm import call_registry

    profile = call_registry.resolve(FEATURE, model=opts.model)
    try:
        connection = call_registry.resolve_provider_connection(profile.provider)
    except ValueError:
        return  # unfamiliar provider shape — let the call itself report
    except call_registry.ProviderConnectionError as exc:
        raise SpecError(
            f"{exc.credential_name}가 설정되어 있지 않다 "
            f"(provider={profile.provider}). credstore를 마운트해 실행하거나 "
            "LLM Gateway를 설정할 것.") from None
    try:
        connection.api_key.encode("ascii")
    except UnicodeEncodeError:
        raise SpecError(
            f"{connection.credential_name} 값에 ASCII가 아닌 문자가 들어 있다. "
            "예시 명령의 자리표시자를 "
            "그대로 붙여넣지 않았는지 확인할 것.") from None


def probe(spec: dict | None = None, opts: Options | None = None) -> list[dict]:
    """Call the provider directly and report what actually came back.

    generate_sync returns None for every failure — bad params, refusal,
    truncation, transport error — so "빈 응답" alone says nothing. This goes
    around it: same profile and params, no exception swallowing, and it
    reports finish_reason and usage so an empty completion can be told apart
    from a rejected request.
    """
    opts = opts or Options()
    from llm import call_registry

    profile = call_registry.resolve(FEATURE, model=opts.model, max_tokens=opts.max_tokens)
    executor = call_registry._EXECUTORS.get(profile.provider)
    if executor is None:
        raise SpecError(f"등록되지 않은 provider: {profile.provider}")

    cases: list[tuple[str, str, str]] = [
        ("minimal", "당신은 번역기다.", "다음을 한국어로: Приказ народного комиссара."),
    ]
    if spec is not None:
        prepared = plan(spec, Options(**{**opts.__dict__, "limit_chunks": 1}))
        chunk = prepared["_chunks"][0]
        body = render_chunk(chunk)
        terms = glossary_for(body, prepared["_glossary"], opts.glossary_limit)
        gloss = "\n".join(f"- {ru} → {ko}" for ru, ko in terms) or "(해당 없음)"
        cases.append((
            f"first-chunk ({len(body):,}자)", SYSTEM_PROMPT,
            f"용어표 (반드시 이 표기를 쓸 것)\n{gloss}\n\n아래 단락들을 번역하라.\n\n{body}",
        ))

    out = []
    for label, system, prompt in cases:
        record: dict = {
            "case": label, "model": profile.model, "provider": profile.provider,
            "maxTokens": profile.max_tokens, "thinking": profile.extra.get("thinking"),
        }
        started = time.time()
        try:
            # The registry executor, not a hand-rolled client: probing a
            # different code path than run() uses is how a "working" probe
            # coexists with a failing run.
            content = executor(profile, prompt, system) or ""
            record.update({
                "ok": bool(content.strip()),
                "contentChars": len(content),
                "preview": content[:200],
            })
        except Exception as e:
            record.update({"ok": False, "contentChars": 0,
                           "error": f"{type(e).__name__}: {e}"})
        record["seconds"] = round(time.time() - started, 1)
        out.append(record)
    return out


def compare(spec: dict, variants: list[str], opts: Options | None = None,
            chunks_wanted: int = 2) -> dict:
    """Translate the same chunks with several models for side-by-side review.

    Picking a model by argument is guesswork; this runs the candidates over
    identical input under the current prompt so the choice rests on the
    output. A variant is "provider/model", optionally "+think" to enable
    DeepSeek reasoning or "+effort=high" for the OpenAI tiers.
    """
    import dataclasses

    from llm import call_registry

    opts = opts or Options()
    prepared = plan(spec, Options(**{**opts.__dict__, "limit_chunks": chunks_wanted}))
    chunks, glossary = prepared["_chunks"], prepared["_glossary"]
    base = call_registry.resolve(FEATURE, model=opts.model, max_tokens=opts.max_tokens)

    results = []
    for variant in variants:
        spec_str, _, flags = variant.partition("+")
        provider, _, model = spec_str.strip().partition("/")
        extra = dict(base.extra)
        if "think" in flags:
            extra["thinking"] = {"type": "enabled"}
        elif provider.startswith("deepseek"):
            extra["thinking"] = {"type": "disabled"}
        if "effort=" in flags:
            extra["reasoning_effort"] = flags.split("effort=")[1].split(",")[0]
        elif provider == "openai":
            extra["reasoning_effort"] = "medium"

        executor = call_registry._EXECUTORS.get(provider)
        if executor is None:
            results.append({"variant": variant, "error": f"unknown provider {provider!r}"})
            continue
        profile = dataclasses.replace(base, provider=provider, model=model or base.model,
                                      extra=extra)

        blocks: dict[int, list[str]] = {}
        problems, seconds, error = [], 0.0, None
        for chunk in chunks:
            prompt = _chunk_prompt(chunk, glossary, opts)
            started = time.time()
            try:
                raw = executor(profile, prompt, SYSTEM_PROMPT) or ""
            except Exception as e:
                error = f"{type(e).__name__}: {e}"
                break
            seconds += time.time() - started
            got = parse_response(raw)
            problems.extend(validate(chunk, got))
            blocks.update(got)
        results.append({
            "variant": variant, "provider": provider, "model": profile.model,
            "thinking": extra.get("thinking"), "effort": extra.get("reasoning_effort"),
            "error": error, "problems": problems, "seconds": round(seconds, 1),
            "blocks": blocks,
        })

    source = {idx: b for chunk in chunks for idx, b in chunk}
    return {"source": source, "results": results,
            "chars": sum(len(l) for b in source.values() for l in b["lines"])}


def plan(spec: dict, opts: Options | None = None) -> dict:
    """Slice, chunk and price a run without calling the model."""
    opts = opts or Options()
    docs = slice_documents(spec)
    glossary = build_glossary(Path(spec["glossary"]["people"]),
                              Path(spec["glossary"]["terms"]),
                              spec["glossary"].get("extra"))
    chunks = [c for d in docs for c in chunk_document(d, opts.max_chars)]
    if opts.limit_chunks:
        chunks = chunks[: opts.limit_chunks]
    total = sum(len(ln) for c in chunks for _, b in c for ln in b["lines"])

    # Price against the model that will actually run, not a hardcoded tier:
    # a stale estimate is worse than none once the call site can change model.
    from llm import call_registry
    from llm.provider_registry import openai_compatible_pricing

    profile = call_registry.resolve(FEATURE, model=opts.model, max_tokens=opts.max_tokens)
    price = openai_compatible_pricing(profile.model)
    thinking_on = (profile.extra.get("thinking") or {}).get("type") == "enabled"
    tokens_in = total / 2.2
    # Reasoning tokens bill as output; high effort roughly doubles it.
    tokens_out = total * 0.9 / 1.6 * (2.0 if thinking_on else 1.0)
    est = tokens_in * price["input"] + tokens_out * price["output"]
    return {
        "id": spec.get("id"),
        "model": profile.model,
        "thinking": thinking_on,
        "documents": [
            {"id": d["id"], "title": d["titleKo"], "blocks": len(d["blocks"]),
             "chars": sum(len(ln) for b in d["blocks"] for ln in b["lines"])}
            for d in docs
        ],
        "glossaryEntries": len(glossary),
        "chunks": len(chunks),
        "chars": total,
        "estimatedUsd": round(est, 4),
        "_docs": docs, "_glossary": glossary, "_chunks": chunks,
    }


def run(spec: dict, opts: Options | None = None,
        progress: Callable[[dict], None] | None = None) -> dict:
    """Translate every chunk and (unless limit_chunks) write the fragment."""
    opts = opts or Options()
    emit = progress or (lambda _e: None)

    # 끝난 스펙은 다시 돌리지 않는다. 조립기는 output을 통째로 다시 쓰므로,
    # 공개된 문서에 스펙이 재현할 수 없는 손질(스펙 밖에서 더한 문서, 손으로
    # 고친 문구)이 들어간 뒤의 재실행은 고침이 아니라 되돌림이다. 새 문헌은
    # 새 스펙으로 옮기면 되고, 이 잠금은 그 길을 막지 않는다.
    if spec.get("frozen"):
        raise SpecError(
            f"{spec.get('id')}: 완료된 스펙이라 재실행이 막혀 있다\n"
            f"  사유: {spec['frozen']}\n"
            f"  출력: {spec.get('output')}\n"
            f"  새 문헌은 새 스펙을 만들어 옮긴다. 이 문서를 정말 다시 만들어야 한다면\n"
            f"  출력 파일과 스펙이 어긋난 곳을 먼저 맞춘 뒤 frozen을 지울 것.")

    prepared = plan(spec, opts)
    docs, glossary, chunks = prepared["_docs"], prepared["_glossary"], prepared["_chunks"]

    cache = Cache(_cache_path(spec, opts.cache_path))
    pending = sum(1 for c in chunks
                  if cache.get(_chunk_key(_chunk_prompt(c, glossary, opts), opts)) is None)
    # Re-assembling a fully cached run (a postEdits tweak, a headnote change)
    # makes no API call, so demanding a credential for it would be wrong.
    if pending:
        preflight(opts)
    emit({"event": "plan", "pending": pending,
          **{k: v for k, v in prepared.items() if not k.startswith("_")}})

    stats = Stats()
    started = time.time()

    with ThreadPoolExecutor(max_workers=opts.concurrency) as pool:
        futures = [pool.submit(_translate_chunk, c, glossary, cache, opts, stats, emit)
                   for c in chunks]
        results, failures = [], []
        for i, (fut, chunk) in enumerate(zip(futures, chunks), 1):
            span = [chunk[0][0], chunk[-1][0]]
            try:
                results.append(fut.result())
            except Exception as e:
                # One bad chunk must not discard the 40 good ones: they are
                # already cached, so a re-run costs only the failures.
                failures.append({"blocks": span, "error": str(e)})
                emit({"event": "chunkFailed", "blocks": span, "error": str(e)})
            emit({"event": "chunk", "done": i, "total": len(futures)})

    translated: dict[int, list[str]] = {}
    for r in results:
        translated.update(r)

    result = {"stats": stats.as_dict(), "seconds": round(time.time() - started, 1),
              "chunks": len(chunks), "failures": failures, "output": None}
    if failures:
        # An incomplete fragment is worse than none: assembly would raise on
        # the first missing block anyway, and a half-written file invites
        # publishing a document with holes in it.
        emit({"event": "done", **result,
              "note": f"{len(failures)}개 청크 실패 — fragment을 쓰지 않았다. "
                      "다시 실행하면 성공한 청크는 캐시에서 나오고 실패분만 재호출된다"})
        return result
    if opts.limit_chunks:
        emit({"event": "done", **result, "note": "limit_chunks — fragment은 쓰지 않았다"})
        return result

    out_path = opts.out_path or Path(spec["output"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    html = assemble(spec, docs, translated)
    out_path.write_text(html, encoding="utf-8")
    result["output"] = str(out_path)
    result["bytes"] = out_path.stat().st_size
    result["strayCyrillic"] = stray_cyrillic(html, spec.get("allowedCyrillic"))
    emit({"event": "done", **result})
    return result
