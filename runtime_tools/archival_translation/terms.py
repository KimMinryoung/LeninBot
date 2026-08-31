"""runtime_tools.archival_translation.terms — LLM 용어 추출과 결정론 집계.

용어 일관성의 판정을 표면 문자열 매칭에서 LLM의 문맥 판단으로 옮긴다.
표면 매칭은 다의어(Союз, Правда, Октябрьский)와 인물 격변화 충돌(Каменева)을
가리지 못했고, 그 오탐을 교정 재시도에 물렸다가 모델의 옳은 첫 번역을
뒤집은 적이 있다(1925 대회 문서). 그래서 여기 있는 두 단계는 어느 쪽도
번역 루프에 관여하지 않는다 — **보고서만 만든다**. 채택(glossary.extra /
glossary.exclude / postEdits)은 사람이 스펙을 고쳐서 한다.

- 사전 스캔(pre): 번역 전에 원문 청크에서 인명·기관·지명·간행물·정치용어를
  문맥 포함(lemma + sense)으로 뽑는다. 용어표 미등재 후보와 제안 표기,
  그리고 표면 매칭으로 걸렸지만 이 문맥에서는 뜻이 다른 용어표 항목
  (misfire → exclude 후보)을 보고한다. 대소문자 신호에 기대지 않으므로
  중국어에도 통한다.
- 사후 감사(post): 번역이 끝난 캐시를 블록 번호로 원문과 정렬해, 항목마다
  번역문이 **실제로 쓴 표기**를 뽑는다. 한 항목에 표기가 둘 이상이면
  불일치, 용어표 항목의 뜻인데 표기가 다르면 이탈로 보고하고 postEdits
  제안을 붙인다. 스펙 postEdits가 이미 덮는 것은 그렇다고 표시한다.

용어표와의 연결도 LLM이 한다: 항목마다 «이 청크에 제시된 용어표 항목 중
같은 것을 가리키는 것»을 ``glossary`` 필드로 답하게 하고, 이탈 판정은 그
연결이 있는 항목에만 한다. 정규식으로 lemma를 용어표에 대면 Союз(나라)가
Союз(의원 그룹) 항목에 걸려 "연방 → 소유즈 (의원 그룹)" 같은 제안이 나온다
— 첫 실행에서 실제로 나왔다.

추출 단위는 번역 청크와 같다(chunk_document). 블록 번호가 곧 마커라 원문·
번역 정렬이 공짜고, 호출 결과는 청크 키로 JSONL에 캐시되어 같은 문서를 다시
보는 데는 호출이 없다. 모델 호출은 registry feature
``archival_term_extraction``(config/llm_call_sites.json) 하나를 지난다.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

from .core import (
    CACHE_DIR,
    Options,
    SourceLanguage,
    _cache_path,
    apply_post_edits,
    glossary_entries_for,
    plan,
)

logger = logging.getLogger(__name__)

FEATURE = "archival_term_extraction"
# 추출 프롬프트·스키마가 바뀌어 옛 캐시의 항목을 같은 것으로 볼 수 없을 때 올린다.
# 2: glossary 연결 필드, sense는 지시체 설명, 상식 지명·기본 어휘 제외, 들 제거.
TERMS_PROMPT_VERSION = "2"

KINDS = ("person", "org", "place", "publication", "term")

SYSTEM_PROMPT = """당신은 사료 번역의 용어 추출기다. 번역하지 않는다. 주어진 단락에서
**표기가 갈릴 수 있는 항목**만 뽑아 JSON 객체 하나로 출력한다.

뽑는 것 (kind)
- person: 인명. 성이 나오면 성을 기준으로 삼고, 이름·부칭만 나오면 그것을 적는다.
- org: 기관·당·단체·회의체와 그 약어 (ЦК, НКВД, Коминтерн, 中央委员会).
- place: 지명. 단, 널리 알려진 나라·대륙·대양 이름(Германия, Франция, Европа,
  Америка, 中国, 苏联)은 표기가 갈릴 여지가 없으므로 뽑지 않는다. 역사적 국명·
  지역명·도시명(Страна Советов, Галиция, Кисловодск)은 뽑는다.
- publication: 신문·잡지·책·문건의 제목.
- term: 제도·정책·사건·이념의 이름과 시대 특유의 정치·경제 용어 (нэп, смычка,
  кулак, план Дауэса, червонец, 修正主义). 기본 정치 어휘(социализм, капитализм,
  буржуазия, пролетариат, класс, революция, 革命)와 일반 어휘는 뽑지 않는다.

항목 필드
- block: 그 항목이 나온 단락의 마커 번호 (정수).
- surface: 원문에 적힌 그대로의 표면형 (곡용된 형태 그대로).
- lemma: 사전형. 러시아어는 주격 단수, 약어는 그대로, 중국어는 surface와 같다.
  같은 지시체는 항상 같은 lemma로 적는다 (Сокольникова, Сокольникову → Сокольников).
- kind: 위 다섯 가지 중 하나.
- sense: 이 항목이 가리키는 대상을 10자 안팎으로 설명한다. 인명은 직위나 소속,
  기관은 정식 명칭, 다의어는 이 문맥의 뜻. 번역 표기나 lemma를 되풀이하지 말고
  대상을 설명할 것 ("소콜니코프"가 아니라 "재무인민위원, 신반대파").
- glossary: 프롬프트에 제시된 «용어표 항목» 목록 중 이 항목과 **같은 것을 가리키는**
  항목의 원문 표기를 그대로 적는다. 뜻이 다르거나 목록에 없으면 null. 표면이 겹쳐도
  뜻이 다르면 null이다 (용어표의 Союз가 '의원 그룹'인데 본문은 Советский Союз →
  null).
- proposed: (사전 스캔 모드) 용어표에 연결되지 않은 항목에 권할 한국어 표기.
  glossary가 있거나 사후 감사 모드면 null.
- target: (사후 감사 모드) 번역문이 이 항목을 옮긴 낱말을 그대로 복사한다.
  조사(은/는/이/가/의/을/를/에게/에서/와/과)와 복수 접미사 '들'은 뗀다. 처음 등장
  때 붙은 원어 병기 괄호는 뺀다. 번역문에 대응하는 표기가 없으면(생략·의역) null.
  사전 스캔 모드면 null.

같은 항목이 한 단락에 여러 번 나와도 단락당 한 번만 적는다. 단락이 다르면 단락마다
적는다.

용어표 오탐 (misfires)
제시된 용어표 항목 중 이 단락들의 문맥에서 **그 뜻으로는 한 번도 쓰이지 않은** 항목의
원문 표기를 misfires에 그대로 적는다. 예: 용어표의 Союз가 '의원 그룹'인데 본문은
Советский Союз(나라)뿐이다 → "Союз". 용어표의 Октябрьский이 성씨인데 본문은
Октябрьская революция뿐이다 → "Октябрьский". 그 뜻으로 한 번이라도 쓰였으면 적지
않는다. 목록에 없는 것도 적지 않는다.

출력 형식 (엄격)
JSON 객체 하나. 다른 텍스트·설명·코드 펜스 없음.
{"terms": [{"block": 12, "surface": "…", "lemma": "…", "kind": "person",
            "sense": "…", "glossary": null, "proposed": null, "target": null}],
 "misfires": ["…"]}"""


# ── 원문·번역 정렬 ─────────────────────────────────────────────────

def source_by_block(docs: list[dict]) -> dict[int, str]:
    """스펙 전역 블록 번호 → 원문 텍스트 (빈 블록 제외)."""
    out: dict[int, str] = {}
    for doc in docs:
        for i, block in enumerate(doc["blocks"]):
            text = "\n".join(block["lines"])
            if text.strip():
                out[doc["offset"] + i] = text
    return out


def align_cached_blocks(
    source_by_idx: dict[int, str], cache_lines: list[str], spec: dict
) -> tuple[list[tuple[str, str]], list[int]]:
    """캐시 JSONL 줄들을 블록 번호로 원문과 정렬해 (원문, 번역) 쌍을 만든다.

    정렬은 캐시 키 재계산이 아니라 **블록 번호**로 한다. 캐시 키에는 시스템
    프롬프트 해시와 청크 옵션이 들어가므로 그중 하나라도 바뀌면 전부 miss가
    되지만, 레코드의 blocks는 스펙 전역 블록 번호로 저장되므로 번호로 직접
    정렬하면 프롬프트·모델·청크 크기가 몇 번을 바뀌었든 복원된다. 파일은
    append-only라 같은 블록이 여러 세대 있으면 뒤(최신) 레코드가 이긴다.
    번역 쪽에는 스펙 postEdits를 적용한다 — TM 적재는 발행본과 같은 텍스트를
    남겨야 한다 (backfill_translation_memory.py가 쓰는 계약).
    """
    target_by_idx = cached_targets(cache_lines)
    pairs: list[tuple[str, str]] = []
    block_ids: list[int] = []
    for idx, target_lines in sorted(target_by_idx.items()):
        source = source_by_idx.get(idx)
        if not source:
            continue
        pairs.append((source, "\n".join(apply_post_edits(target_lines, spec))))
        block_ids.append(idx)
    return pairs, block_ids


def cached_targets(cache_lines: list[str]) -> dict[int, list[str]]:
    """캐시 JSONL 줄들 → 블록 번호별 모델 원출력 (postEdits 미적용, 최신 우선)."""
    out: dict[int, list[str]] = {}
    for line in cache_lines:
        if not line.strip():
            continue
        rec = json.loads(line)
        for key, lines in (rec.get("blocks") or {}).items():
            if lines:
                out[int(key)] = lines
    return out


def translated_blocks(spec: dict, docs: list[dict], lang: SourceLanguage,
                      cache_path: Path | None = None) -> tuple[dict[int, list[str]], list[int]]:
    """사후 감사가 볼 번역문: 청크 캐시(모델 원출력)를 블록 번호로 모으고, 캐시에
    없는 블록(TM prefill로 채워진 것)은 TM에서 보충한다. 원출력을 보는 이유는
    postEdits가 무엇을 이미 고쳤는지를 보고서가 따로 표시하기 위해서다.
    반환: (블록 → 줄들, 번역을 못 찾은 원문 블록 번호들)."""
    path = cache_path or _cache_path(spec, None)
    lines = path.read_text(encoding="utf-8").splitlines() if path.is_file() else []
    targets = cached_targets(lines)
    sources = source_by_block(docs)
    missing = [idx for idx in sources if idx not in targets]
    if missing:
        try:
            from runtime_tools import translation_memory
            hits = translation_memory.exact_matches(
                [sources[i] for i in missing], lang_pair=f"{lang.code}-ko")
            for idx in list(missing):
                text = sources[idx].strip()
                if text in hits:
                    targets[idx] = hits[text].split("\n")
                    missing.remove(idx)
        except Exception as e:  # TM은 보충 수단일 뿐, 없어도 감사는 돈다
            logger.warning("[archival-terms] TM 보충 실패: %s", e)
    return targets, sorted(missing)


# ── 프롬프트 ─────────────────────────────────────────────────────────

def _offered_text(offered: list[dict]) -> str:
    body = "\n".join(f"- {g['ru']} → {g['ko']}" for g in offered) or "(해당 없음)"
    return ("용어표 항목 (이 단락들에 표면 매칭으로 걸린 것 — 같은 것을 가리키면 glossary에, "
            f"그 뜻으로 한 번도 안 쓰였으면 misfires에)\n{body}\n\n")


def render_pre_prompt(chunk, offered: list[dict]) -> str:
    body = "\n\n".join(
        f"[[{idx}|{b['tag']}]]\n" + "\n".join(b["lines"]) for idx, b in chunk if b["lines"])
    return (_offered_text(offered)
            + "모드: 사전 스캔 (번역 전). proposed를 채우고 target은 null로 둔다.\n\n"
            + body)


def render_post_prompt(chunk, offered: list[dict], targets: dict[int, list[str]]) -> str:
    parts = []
    for idx, b in chunk:
        if not b["lines"] or idx not in targets:
            continue
        parts.append(f"[[{idx}|{b['tag']}]]\n원문: " + "\n".join(b["lines"])
                     + "\n번역: " + "\n".join(targets[idx]))
    return (_offered_text(offered)
            + "모드: 사후 감사. 각 항목에 번역문이 실제 쓴 표기를 target에 적는다. "
              "proposed는 null.\n\n"
            + "\n\n".join(parts))


_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.M)
# 표기 끝에 붙어 온 원어 병기 괄호: 사르키스(Саркис), 중앙위원회(ЦК), 캉성(康生).
# 괄호 안이 한국어뿐인 것 — 전연방공산당(볼셰비키) — 은 이름의 일부라 남긴다.
_GLOSS_TAIL_RE = re.compile(
    r"\s*[(（][^)）]*[А-Яа-яЁёІіЇїЄєA-Za-z㐀-䶿一-鿿][^)）]*[)）]\s*$")


def _clean_target(value) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    text = _GLOSS_TAIL_RE.sub("", text).strip()
    text = re.sub(r"(?<=[가-힣])들$", "", text)
    return text or None


def parse_extraction(raw: str) -> dict | None:
    """모델 응답 → {"terms": [...], "misfires": [...]} 또는 None(못 읽음).

    항목은 필드 형식을 맞춰 정리한다: block은 정수, kind는 KINDS 안, surface·
    lemma는 비어 있지 않아야 남는다. target에서는 모델이 지시를 어기고 붙여
    온 원어 병기 괄호와 복수 접미사를 결정론으로 뗀다 — 표기 비교의 잡음일
    뿐 판정 대상이 아니다. 모델이 코드 펜스를 두르거나 앞뒤에 말을 붙여도
    첫 '{'부터 마지막 '}'까지를 읽는다."""
    text = _FENCE_RE.sub("", (raw or "").strip())
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        data = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    terms = []
    for t in data.get("terms") or []:
        if not isinstance(t, dict):
            continue
        try:
            block = int(t.get("block"))
        except (TypeError, ValueError):
            continue
        surface = str(t.get("surface") or "").strip()
        lemma = str(t.get("lemma") or "").strip() or surface
        if not surface:
            continue
        kind = str(t.get("kind") or "term").strip().lower()
        terms.append({
            "block": block, "surface": surface, "lemma": lemma,
            "kind": kind if kind in KINDS else "term",
            "sense": str(t.get("sense") or "").strip(),
            "glossary": (str(t["glossary"]).strip() or None) if t.get("glossary") else None,
            "proposed": (str(t["proposed"]).strip() or None) if t.get("proposed") else None,
            "target": _clean_target(t.get("target")),
        })
    misfires = [str(m).strip() for m in (data.get("misfires") or []) if str(m).strip()]
    return {"terms": terms, "misfires": misfires}


# ── 호출과 캐시 ──────────────────────────────────────────────────────

class _JsonlCache:
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

    def put(self, key: str, rec: dict) -> None:
        rec = {"key": key, **rec}
        with self.lock:
            self.data[key] = rec
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def terms_cache_path(spec: dict) -> Path:
    return CACHE_DIR / f"{spec.get('id', 'unnamed')}.terms.jsonl"


def _profile():
    from llm import call_registry
    return call_registry.resolve(FEATURE)


def _chunk_key(mode: str, prompt: str) -> str:
    p = _profile()
    return hashlib.sha256(
        f"{TERMS_PROMPT_VERSION}\0{mode}\0{p.provider}\0{p.model}\0{prompt}".encode("utf-8")
    ).hexdigest()


def _call(prompt: str) -> str | None:
    """registry 경유 단발 호출. 테스트는 이 함수를 바꿔 끼운다."""
    from llm import call_registry
    return call_registry.generate_sync(FEATURE, prompt, system=SYSTEM_PROMPT)


def extract(chunks, mode: str, glossary: list[dict], opts: Options,
            cache: _JsonlCache, *, targets: dict[int, list[str]] | None = None,
            emit: Callable[[dict], None] | None = None,
            concurrency: int = 4) -> list[dict]:
    """청크마다 추출 호출을 하고 레코드 목록을 돌려준다.

    레코드: {"blocks": [첫, 끝], "offered": [원문 표기…], "terms": [...],
             "misfires": [...], "error": str|None, "cached": bool}
    항목의 glossary 연결은 그 청크에 실제로 제시된 항목일 때만 남긴다(프롬프트
    위반은 버린다). JSON을 못 읽으면 한 번 더 부르고, 그래도 안 되면 error
    레코드로 남긴 채 다음 청크로 간다 — 한 청크의 실패가 문서 전체 보고를
    막으면 안 된다."""
    emit = emit or (lambda _e: None)
    if mode not in ("pre", "post"):
        raise ValueError(f"mode must be pre|post, got {mode!r}")

    def one(chunk) -> dict:
        body = "\n".join(ln for _, b in chunk for ln in b["lines"])
        offered = glossary_entries_for(body, glossary, opts.glossary_limit)
        offered_ru = [g["ru"] for g in offered]
        if mode == "pre":
            prompt = render_pre_prompt(chunk, offered)
        else:
            prompt = render_post_prompt(chunk, offered, targets or {})
        span = [chunk[0][0], chunk[-1][0]]
        base = {"blocks": span, "offered": offered_ru}
        if not any(b["lines"] and (mode == "pre" or idx in (targets or {}))
                   for idx, b in chunk):
            return {**base, "terms": [], "misfires": [], "error": None, "cached": True}
        key = _chunk_key(mode, prompt)
        hit = cache.get(key)
        if hit:
            return {**base, "terms": hit["terms"], "misfires": hit["misfires"],
                    "error": None, "cached": True}
        parsed, err = None, "빈 응답"
        for attempt in (1, 2):
            raw = _call(prompt)
            parsed = parse_extraction(raw or "")
            if parsed is not None:
                break
            err = "빈 응답" if not raw else f"JSON을 읽을 수 없음: {raw[:80]!r}"
            emit({"event": "extractRetry", "blocks": span, "attempt": attempt, "error": err})
            time.sleep(1.5 * attempt)
        if parsed is None:
            emit({"event": "extractFailed", "blocks": span, "error": err})
            return {**base, "terms": [], "misfires": [], "error": err, "cached": False}
        offered_set = set(offered_ru)
        for t in parsed["terms"]:
            if t["glossary"] not in offered_set:
                t["glossary"] = None
        cache.put(key, {"mode": mode, "blocks": span, **parsed})
        return {**base, **parsed, "error": None, "cached": False}

    records: list[dict] = []
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        futures = [pool.submit(one, c) for c in chunks]
        for i, fut in enumerate(futures, 1):
            records.append(fut.result())
            emit({"event": "chunk", "done": i, "total": len(futures)})
    return records


# ── 결정론 집계 ──────────────────────────────────────────────────────

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip()).lower()


def match_glossary(lemma: str, surfaces: list[str], glossary: list[dict]) -> dict | None:
    """lemma(우선)나 표면형이 용어표 항목 하나에 **통째로** 걸리면 그 항목.

    사전 스캔에서 '이미 등재된 항목'을 후보에서 거르는 데만 쓴다 — 표면
    일치라 다의어를 못 가리므로 이탈 판정에는 쓰지 않는다(그건 LLM의
    glossary 연결이 한다). 부분 일치는 다른 용어다(Центральный Комитет
    партии ≠ Центральный Комитет). 패턴은 build_glossary의 것을 그대로
    쓰므로 러시아어 곡용 변형과 경계 가드가 같이 적용된다."""
    for text in [lemma, *surfaces]:
        text = (text or "").strip()
        if not text:
            continue
        for g in glossary:
            m = g["pattern"].search(text)
            if m and m.span() == (0, len(text)):
                return g
    return None


def aggregate(records: list[dict]) -> list[dict]:
    """청크 레코드들 → lemma별 그룹.

    그룹: {"lemma", "kind"(최빈), "kinds": Counter, "surfaces": [...],
           "senses": Counter, "blocks": [...], "count": n, "proposed": Counter,
           "links": Counter(용어표 원문 표기 → 횟수),
           "targets": {표기: [블록…]}, "targetSenses": {표기: [sense…]},
           "targetLinks": {표기: {용어표 표기: [블록…]}}, "missing": [target 없는 블록…]}
    묶는 키는 lemma 소문자뿐이다. kind는 같은 지시체에도 청크마다 흔들리고
    (Советский Союз가 place와 org로 갈라졌다), sense는 표현이 흔들려서 키로
    쓰면 같은 인물이 여러 그룹으로 흩어진다 — 둘 다 집계로만 남긴다."""
    groups: dict[str, dict] = {}
    for rec in records:
        for t in rec.get("terms") or []:
            key = _norm(t["lemma"])
            g = groups.get(key)
            if g is None:
                g = groups[key] = {
                    "lemma": t["lemma"], "kinds": Counter(), "surfaces": [],
                    "senses": Counter(), "blocks": [], "count": 0,
                    "proposed": Counter(), "links": Counter(),
                    "targets": defaultdict(list), "targetSenses": defaultdict(set),
                    "targetLinks": defaultdict(lambda: defaultdict(list)), "missing": [],
                }
            g["count"] += 1
            g["kinds"][t["kind"]] += 1
            g["blocks"].append(t["block"])
            if t["surface"] not in g["surfaces"]:
                g["surfaces"].append(t["surface"])
            if t["sense"]:
                g["senses"][t["sense"]] += 1
            if t.get("proposed"):
                g["proposed"][t["proposed"]] += 1
            if t.get("glossary"):
                g["links"][t["glossary"]] += 1
            if t.get("target"):
                g["targets"][t["target"]].append(t["block"])
                if t["sense"]:
                    g["targetSenses"][t["target"]].add(t["sense"])
                if t.get("glossary"):
                    g["targetLinks"][t["target"]][t["glossary"]].append(t["block"])
            else:
                g["missing"].append(t["block"])
    out = []
    for g in groups.values():
        g["kind"] = g["kinds"].most_common(1)[0][0]
        g["blocks"] = sorted(set(g["blocks"]))
        g["targets"] = {k: sorted(set(v)) for k, v in g["targets"].items()}
        g["targetSenses"] = {k: sorted(v) for k, v in g["targetSenses"].items()}
        g["targetLinks"] = {k: {ru: sorted(set(b)) for ru, b in v.items()}
                            for k, v in g["targetLinks"].items()}
        g["missing"] = sorted(set(g["missing"]))
        out.append(g)
    out.sort(key=lambda g: (-g["count"], g["lemma"]))
    return out


def misfire_report(records: list[dict]) -> list[dict]:
    """용어표 항목별 (제시된 청크 수, 오탐으로 돌아온 청크 수).

    전 청크에서 오탐이면 glossary.exclude 후보다. 일부에서만 오탐이면 이
    문서 안에서 두 뜻으로 쓰이는 것이라 exclude가 아니라 sense 분리 대상이고,
    그 구분을 사람이 할 수 있게 비율을 그대로 보인다."""
    offered: Counter = Counter()
    flagged: Counter = Counter()
    for rec in records:
        for ru in rec.get("offered") or []:
            offered[ru] += 1
        offered_set = set(rec.get("offered") or [])
        for ru in set(rec.get("misfires") or []):
            if ru in offered_set:  # 목록에 없는 것을 적으면 무시 (프롬프트 위반)
                flagged[ru] += 1
    out = [{"ru": ru, "offered": offered[ru], "misfired": n,
            "always": n == offered[ru]}
           for ru, n in flagged.items()]
    out.sort(key=lambda r: (-r["misfired"] / max(r["offered"], 1), -r["misfired"], r["ru"]))
    return out


def _first_context(surface: str, blocks: list[int], sources: dict[int, str]) -> str:
    for idx in blocks:
        text = sources.get(idx, "")
        pos = text.find(surface)
        if pos >= 0:
            start = max(0, pos - 40)
            return text[start:pos + len(surface) + 60].replace("\n", " ").strip()
    return (sources.get(blocks[0], "") if blocks else "")[:100].replace("\n", " ")


def _linked_entry(g: dict, glossary: list[dict]) -> dict | None:
    """LLM이 이 그룹의 항목들을 가장 자주 연결한 용어표 항목."""
    if not g["links"]:
        return None
    ru = g["links"].most_common(1)[0][0]
    return next((e for e in glossary if e["ru"] == ru), None)


def pre_report(records: list[dict], glossary: list[dict], sources: dict[int, str],
               *, min_count: int = 2) -> dict:
    """사전 스캔 보고: 미등재 후보(제안 표기 포함)와 용어표 오탐 후보.

    등재 여부는 LLM 연결 또는 표면 전체 일치 중 하나라도 있으면 등재로 본다 —
    여기서 표면 일치의 오탐은 후보 하나를 덜 보여줄 뿐 잘못된 제안을 만들지
    않는다."""
    candidates = []
    registered = []
    for g in aggregate(records):
        entry = _linked_entry(g, glossary) or match_glossary(g["lemma"], g["surfaces"], glossary)
        if entry is not None:
            registered.append({"lemma": g["lemma"], "kind": g["kind"], "count": g["count"],
                               "glossary": entry["ru"], "ko": entry["ko"]})
            continue
        if g["count"] < min_count:
            continue
        proposed = g["proposed"].most_common(1)[0][0] if g["proposed"] else None
        candidates.append({
            "lemma": g["lemma"], "kind": g["kind"], "count": g["count"],
            "surfaces": g["surfaces"], "senses": [s for s, _ in g["senses"].most_common(3)],
            "proposed": proposed,
            "proposedAll": [p for p, _ in g["proposed"].most_common()],
            "blocks": g["blocks"],
            "context": _first_context(g["surfaces"][0], g["blocks"], sources),
        })
    misfires = misfire_report(records)
    extra_snippet = {c["lemma"]: c["proposed"] for c in candidates if c["proposed"]}
    exclude_snippet = [m["ru"] for m in misfires if m["always"]]
    failed = [r["blocks"] for r in records if r.get("error")]
    return {"candidates": candidates, "registered": registered, "misfires": misfires,
            "extraSnippet": extra_snippet, "excludeSnippet": exclude_snippet,
            "failedChunks": failed,
            "chunks": len(records), "cachedChunks": sum(1 for r in records if r.get("cached"))}


def _span_mismatch(a: str, b: str) -> bool:
    """한 표기가 다른 표기를 통째로 품으면 표기 차이가 아니라 추출 범위 차이다.

    '안정화' vs '자본주의의 안정화', '러시아공산당(볼셰비키)' vs '러시아공산당
    (볼셰비키) 중앙위원회' — LLM이 용어표 항목보다 넓거나 좁은 구절을 lemma로
    잡은 것이지 다른 번역을 쓴 것이 아니다. 이런 쌍을 postEdits로 치환하면
    문서를 망가뜨린다(짧은 쪽이 키가 되어 긴 쪽 안에서도 치환된다)."""
    x, y = _norm(a), _norm(b)
    return x != y and (x in y or y in x)


_KO_ENDINGS = {
    "", "는", "은", "이", "가", "을", "를", "의", "과", "와", "로", "으로", "에", "에서", "도", "만",
    "인", "이고", "이다", "이며", "이라", "이란", "이라는", "이었다", "였다", "다", "다고", "라고",
    "한", "하고", "하는", "하며", "한다", "했다", "할", "된", "되고", "되는", "된다", "되었다", "될",
    "적", "적인", "적으로", "들",
}


def _inflection_variant(a: str, b: str) -> bool:
    """두 표기가 어미·조사만 다른 같은 말이면 True: '법적으로 무효인' / '법적으로
    무효이고', '노농동맹을' / '노농동맹'. 한국어는 낱말 끝에서 굴절하므로, 공통
    앞부분 뒤에 남는 꼬리가 둘 다 어미·조사 목록 안에 있으면 표기가 다른 게 아니라
    문장에 맞춰 활용한 것이다. 꼬리를 목록으로 제한하는 이유: 그냥 '끝 두 글자
    차이'로 잡으면 '카메네바'/'카메네프' 같은 음차 차이까지 같은 말이 된다.
    용어표의 구 단위 항목(«недействительными с момента их подписания» → '서명한
    순간부터 효력이 없는')이 문장 속에서 '…효력이 없다고'로 나온 것을 이탈로 제안하면
    postEdits가 문법을 부순다."""
    x, y = _norm(a), _norm(b)
    if x == y or not x or not y:
        return False
    common = 0
    for cx, cy in zip(x, y):
        if cx != cy:
            break
        common += 1
    if common < 2:
        return False
    tail_x, tail_y = x[common:].strip(), y[common:].strip()
    return tail_x in _KO_ENDINGS and tail_y in _KO_ENDINGS


def _covered_by_post_edits(target: str, blocks: list[int],
                           targets: dict[int, list[str]], spec: dict) -> list[int]:
    """postEdits를 적용하면 그 표기가 사라지는(줄어드는) 블록들.

    postEdits 키는 조사나 앞말이 붙은 구절('지노비예프와 카메네바')일 수
    있으므로 표기 낱말 자체가 아니라 블록 텍스트에 적용해 본 결과로 판단한다."""
    covered = []
    for idx in blocks:
        raw = "\n".join(targets.get(idx, []))
        edited = "\n".join(apply_post_edits(targets.get(idx, []), spec))
        if raw.count(target) > edited.count(target):
            covered.append(idx)
    return covered


def post_report(records: list[dict], glossary: list[dict], spec: dict,
                targets: dict[int, list[str]], *, min_count: int = 1) -> dict:
    """사후 감사 보고: 표기 불일치, 용어표 이탈, postEdits 제안.

    기준 표기(canonical)는 (1) LLM이 연결한 용어표 항목의 표기, 없으면 (2)
    postEdits가 덮고 남은 블록이 가장 많은 표기(동률이면 첫 등장)다. 첫
    실행에서 배운 두 가지가 여기 반영돼 있다: 다수 표기가 이미 postEdits로
    고쳐진 오역이면("소비에트 소유즈" ×2) 그것을 기준으로 삼으면 안 되고,
    용어표 항목이 있으면 불일치 제안과 이탈 제안이 서로 반대 방향("라린→
    루리예"와 "루리예→라린")으로 나와서는 안 된다."""
    inconsistent, deviations = [], []
    suggestions: dict[str, str] = {}
    for g in aggregate(records):
        if g["count"] < min_count or not g["targets"]:
            continue
        entry = _linked_entry(g, glossary)
        cover = {t: _covered_by_post_edits(t, blocks, targets, spec)
                 for t, blocks in g["targets"].items()}
        remaining = {t: [b for b in blocks if b not in cover[t]]
                     for t, blocks in g["targets"].items()}
        # 남은 블록 수, 동률이면 첫 등장. 총 블록 수는 보지 않는다 — postEdits가
        # 이미 고친 등장은 그 표기가 틀렸다는 증거지 다수라는 증거가 아니다.
        ranked = sorted(g["targets"].items(),
                        key=lambda kv: (-len(remaining[kv[0]]), min(kv[1])))
        if entry is not None:
            canonical = entry["ko"]
        else:
            canonical = ranked[0][0] if any(remaining.values()) else None
        canon_senses = set(g["targetSenses"].get(canonical, [])) if canonical else set()

        def variant(t, blocks):
            senses = set(g["targetSenses"].get(t, []))
            disjoint = bool(senses and canon_senses and not (senses & canon_senses))
            linked = bool(entry and g["targetLinks"].get(t, {}).get(entry["ru"]))
            return {"target": t, "blocks": blocks, "covered": cover[t],
                    "remaining": remaining[t], "senses": sorted(senses),
                    "senseDisjoint": disjoint, "linked": linked,
                    "spanMismatch": bool(canonical) and _span_mismatch(t, canonical),
                    "inflection": bool(canonical) and _inflection_variant(t, canonical)}

        def suggest(v):
            if not v["remaining"] or canonical is None or _norm(v["target"]) == _norm(canonical):
                return
            if v["spanMismatch"] or v["inflection"]:
                return
            # 용어표 항목이 있으면 그 항목에 연결된 표기만 제안한다 — 연결이 없는
            # 것은 LLM이 다른 대상으로 본 것이다(Союз의 '연방'). 항목이 없으면
            # sense가 기준 표기와 전혀 안 겹치는 것(совет의 '충고')만 뺀다.
            if entry is not None and not v["linked"]:
                return
            if entry is None and v["senseDisjoint"]:
                return
            suggestions.setdefault(v["target"], canonical)

        if len(ranked) > 1:
            majority, majority_blocks = ranked[0]
            variants = [variant(t, b) for t, b in ranked[1:]]
            for v in variants:
                suggest(v)
            inconsistent.append({
                "lemma": g["lemma"], "kind": g["kind"], "count": g["count"],
                "canonical": canonical,
                "majority": majority, "majorityBlocks": majority_blocks,
                "majorityCovered": cover[majority],
                "majoritySenses": sorted(g["targetSenses"].get(majority, [])),
                "glossary": (entry["ru"], entry["ko"]) if entry else None,
                "variants": variants,
            })

        if entry is not None:
            for t, blocks in ranked:
                if _norm(t) == _norm(entry["ko"]):
                    continue
                linked_blocks = g["targetLinks"].get(t, {}).get(entry["ru"])
                if not linked_blocks:
                    continue  # 이 표기의 등장은 그 용어표 항목에 연결되지 않았다
                covered = [b for b in linked_blocks if b in cover[t]]
                rem = [b for b in linked_blocks if b not in cover[t]]
                span = _span_mismatch(t, entry["ko"])
                inflected = _inflection_variant(t, entry["ko"])
                deviations.append({
                    "lemma": g["lemma"], "kind": g["kind"],
                    "glossary": entry["ru"], "expected": entry["ko"], "target": t,
                    "blocks": linked_blocks, "covered": covered, "remaining": rem,
                    "senses": g["targetSenses"].get(t, []), "spanMismatch": span,
                    "inflection": inflected,
                })
                if rem and not span and not inflected:
                    suggestions.setdefault(t, entry["ko"])

    failed = [r["blocks"] for r in records if r.get("error")]
    return {"inconsistent": inconsistent, "deviations": deviations,
            "postEditsSnippet": suggestions, "failedChunks": failed,
            "chunks": len(records), "cachedChunks": sum(1 for r in records if r.get("cached"))}


# ── 마크다운 ─────────────────────────────────────────────────────────

def _blocks(ids: list[int], limit: int = 12) -> str:
    shown = ", ".join(str(i) for i in ids[:limit])
    return shown + (f" … 외 {len(ids) - limit}" if len(ids) > limit else "")


def render_pre_markdown(spec: dict, report: dict) -> str:
    out = [f"# 용어 사전 스캔 (LLM) — {spec.get('id')}", "",
           f"청크 {report['chunks']}개 (캐시 {report['cachedChunks']}), "
           f"미등재 후보 {len(report['candidates'])}건, 오탐 의심 {len(report['misfires'])}건, "
           f"등재 확인 {len(report['registered'])}건", ""]
    if report["failedChunks"]:
        out += ["> 추출 실패 청크: " + "; ".join(f"{a}–{b}" for a, b in report["failedChunks"]), ""]
    out += ["## 용어표 미등재 후보", "",
            "| 횟수 | kind | lemma | 표면형 | sense | 제안 표기 | 문맥 |", "|---|---|---|---|---|---|---|"]
    for c in report["candidates"]:
        out.append(f"| {c['count']} | {c['kind']} | {c['lemma']} | {', '.join(c['surfaces'][:4])} "
                   f"| {' / '.join(c['senses'])} | {c['proposed'] or ''} | {c['context']} |")
    out += ["", "## 용어표 오탐 의심 (misfires)", "",
            "| 항목 | 제시 청크 | 오탐 청크 | 판단 |", "|---|---|---|---|"]
    for m in report["misfires"]:
        verdict = "전 청크 오탐 → exclude 후보" if m["always"] else "일부만 오탐 → 이 문서에서 두 뜻으로 쓰임"
        out.append(f"| {m['ru']} | {m['offered']} | {m['misfired']} | {verdict} |")
    out += ["", "## 붙여넣을 조각", "", "`glossary.extra` (제안 표기는 검토 후 채택):", "```json",
            json.dumps(report["extraSnippet"], ensure_ascii=False, indent=2), "```", "",
            "`glossary.exclude`:", "```json",
            json.dumps(report["excludeSnippet"], ensure_ascii=False), "```", ""]
    return "\n".join(out)


def _coverage_note(v: dict) -> list[str]:
    if v["covered"] and not v["remaining"]:
        return ["postEdits 적용됨"]
    if v["covered"]:
        return [f"postEdits가 {len(v['covered'])}블록만 덮음, 남은 블록 {_blocks(v['remaining'])}"]
    return []


def render_post_markdown(spec: dict, report: dict, missing_blocks: list[int]) -> str:
    out = [f"# 용어 사후 감사 (LLM) — {spec.get('id')}", "",
           f"청크 {report['chunks']}개 (캐시 {report['cachedChunks']}), "
           f"표기 불일치 {len(report['inconsistent'])}건, 용어표 이탈 {len(report['deviations'])}건", ""]
    if missing_blocks:
        out += [f"> 번역을 찾지 못한 원문 블록 {len(missing_blocks)}개: {_blocks(missing_blocks)}", ""]
    if report["failedChunks"]:
        out += ["> 추출 실패 청크: " + "; ".join(f"{a}–{b}" for a, b in report["failedChunks"]), ""]
    out += ["## 표기 불일치 (한 항목에 표기가 둘 이상)", "",
            "기준 표기 = 용어표 표기, 없으면 postEdits가 덮고 남은 블록이 가장 많은 표기.", ""]
    for item in report["inconsistent"]:
        gl = f" — 용어표 {item['glossary'][0]} → {item['glossary'][1]}" if item["glossary"] else ""
        out.append(f"### {item['lemma']} ({item['kind']}, {item['count']}회){gl}")
        out.append(f"- 기준: **{item['canonical'] or '(없음 — 전부 postEdits로 덮임)'}**")
        maj_note = " (postEdits 적용됨)" if item["majorityCovered"] and len(item["majorityCovered"]) == len(item["majorityBlocks"]) else ""
        out.append(f"- 다수: **{item['majority']}** ×{len(item['majorityBlocks'])} "
                   f"[{_blocks(item['majorityBlocks'])}]{maj_note}"
                   + (f" — sense: {' / '.join(item['majoritySenses'])}" if item["majoritySenses"] else ""))
        for v in item["variants"]:
            note = _coverage_note(v)
            if v["senseDisjoint"]:
                note.append("sense 상이 — 다의어일 수 있음")
            if item["glossary"] and not v["linked"]:
                note.append("용어표 항목에 연결되지 않음 — 다른 대상일 수 있음")
            if v["spanMismatch"]:
                note.append("범위 차이 — 한쪽이 다른 쪽을 품음, 제안 제외")
            if v["inflection"]:
                note.append("어미·조사 차이 — 같은 표기, 제안 제외")
            sense = f" — sense: {' / '.join(v['senses'])}" if v["senses"] else ""
            out.append(f"- 소수: **{v['target']}** ×{len(v['blocks'])} [{_blocks(v['blocks'])}]{sense}"
                       + (f" ({'; '.join(note)})" if note else ""))
        out.append("")
    out += ["## 용어표 이탈 (LLM이 용어표 항목에 연결했는데 표기가 다름)", "",
            "| lemma | 용어표 | 기대 표기 | 실제 표기 | 블록 | 상태 |", "|---|---|---|---|---|---|"]
    for d in report["deviations"]:
        if d["covered"] and not d["remaining"]:
            status = "postEdits 적용됨"
        elif d["spanMismatch"]:
            status = "범위 차이(추출 구절이 용어표 항목보다 넓거나 좁음) — 제안 제외"
        elif d["inflection"]:
            status = "어미·조사 차이 — 같은 표기, 제안 제외"
        elif d["covered"]:
            status = f"일부만 적용, 남은 블록 {_blocks(d['remaining'])}"
        else:
            status = "미처리"
        sense = f" ({' / '.join(d['senses'])})" if d["senses"] else ""
        out.append(f"| {d['lemma']}{sense} | {d['glossary']} | {d['expected']} | {d['target']} "
                   f"| {_blocks(d['blocks'])} | {status} |")
    out += ["", "## postEdits 제안 (미처리분만; 용어표 미연결·다의어 의심·범위 차이는 제외)", "```json",
            json.dumps(report["postEditsSnippet"], ensure_ascii=False, indent=2), "```", "",
            "제안은 검토용이다 — 키가 짧은 낱말이면 다른 문맥까지 치환되므로 반드시 원문을 확인하고, "
            "필요하면 앞뒤 말을 붙여 좁힌다. 적용은 스펙 `postEdits`에 넣고 재실행한다 — 전 청크 "
            "캐시 적중이라 모델 호출 없이 재조립된다.", ""]
    return "\n".join(out)


# ── 진입점 ───────────────────────────────────────────────────────────

def estimate(spec: dict, opts: Options, mode: str) -> dict:
    """호출 없이 청크 수·예상 비용. 사후는 원문+번역이 함께 가므로 입력이 두 배쯤."""
    from llm.provider_registry import openai_compatible_pricing

    prepared = plan(spec, opts)
    lang = prepared["_lang"]
    profile = _profile()
    price = openai_compatible_pricing(profile.model)
    chars = prepared["chars"]
    tokens_in = chars / lang.chars_per_token * (1 + lang.output_ratio if mode == "post" else 1)
    tokens_in += len(prepared["_chunks"]) * 900  # 시스템 프롬프트 + 용어표 목록
    tokens_out = len(prepared["_chunks"]) * 1500
    cache = _JsonlCache(terms_cache_path(spec))
    return {"id": spec.get("id"), "mode": mode, "sourceLang": lang.code,
            "model": profile.model, "provider": profile.provider,
            "chunks": prepared["chunks"], "chars": chars,
            "cachedRecords": sum(1 for r in cache.data.values() if r.get("mode") == mode),
            "estimatedUsd": round(tokens_in * price["input"] + tokens_out * price["output"], 4)}


def pre_scan(spec: dict, opts: Options | None = None, *, min_count: int = 2,
             emit: Callable[[dict], None] | None = None, concurrency: int = 4) -> dict:
    opts = opts or Options()
    prepared = plan(spec, opts)
    cache = _JsonlCache(terms_cache_path(spec))
    records = extract(prepared["_chunks"], "pre", prepared["_glossary"], opts, cache,
                      emit=emit, concurrency=concurrency)
    report = pre_report(records, prepared["_glossary"], source_by_block(prepared["_docs"]),
                        min_count=min_count)
    return {**report, "markdown": render_pre_markdown(spec, report), "records": records}


def post_audit(spec: dict, opts: Options | None = None, *, min_count: int = 1,
               emit: Callable[[dict], None] | None = None, concurrency: int = 4) -> dict:
    opts = opts or Options()
    prepared = plan(spec, opts)
    targets, missing = translated_blocks(spec, prepared["_docs"], prepared["_lang"],
                                         opts.cache_path)
    if not targets:
        raise RuntimeError(f"{spec.get('id')}: 번역 캐시가 없다 — 먼저 번역을 돌릴 것")
    cache = _JsonlCache(terms_cache_path(spec))
    records = extract(prepared["_chunks"], "post", prepared["_glossary"], opts, cache,
                      targets=targets, emit=emit, concurrency=concurrency)
    report = post_report(records, prepared["_glossary"], spec, targets, min_count=min_count)
    return {**report, "missingBlocks": missing,
            "markdown": render_post_markdown(spec, report, missing), "records": records}
