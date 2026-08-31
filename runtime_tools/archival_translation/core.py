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
import sys
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

# 모델 라우팅은 언어쌍별이다: RU→KO와 ZH→KO는 서로 다른 registry feature를
# 쓴다(SourceLanguage.feature). 어느 쌍의 모델을 바꾸려면
# config/llm_call_sites.json의 해당 항목을 고치거나
# LLM_SITE_ARCHIVAL_DOCUMENT_TRANSLATION_{RU,ZH}_MODEL 환경변수를 쓴다.
# 교체 전에 --compare로 같은 청크를 후보 모델들로 나란히 뽑아 확인할 것.
# 캐시 키에는 시스템 프롬프트 해시와 유저 프롬프트 전문이 이미 들어간다
# (_chunk_key). 검증기(validate)가 엄격해진 경우는 이 상수를 올리지 않는다 —
# 캐시에서 꺼낸 청크도 현재 검증기를 다시 통과해야 쓰이므로(_cached_blocks),
# 새 검사에 걸리는 청크만 다시 번역되고 나머지 캐시는 그대로 산다. 이 상수는
# 파서(parse_response)가 바뀌어 옛 레코드의 블록 분할 자체를 믿을 수 없게
# 됐을 때만 올린다.
PROMPT_VERSION = "2"

_SPEC_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
# Any tag an adapter emits, not a fixed list. Pinning it to militera's tags
# made every wikisource center/dd/li block read as a missing marker — the
# model had returned them correctly and the parser refused to see them.
MARKER_RE = re.compile(r"\[\[(\d+)\|([a-z][a-z0-9]*)\]\][ \t]*\n?")
CYRILLIC_RE = re.compile(r"[а-яА-ЯёЁіІїЇєЄ]")
HAN_RE = re.compile(r"[㐀-䶿一-鿿]")
LATIN_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]")
HANGUL_RE = re.compile(r"[가-힣]")
# 원문/번역문의 "문장이 끝났다"는 신호. validate()의 끊김 검사가 쓴다.
_SENTENCE_END_SRC = re.compile(r'[.!?…»”"\)\]。！？」』]$')
_SENTENCE_END_TGT = re.compile(r'[.!?…。」』”"\)\]:;]$')

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
- 번역투 금지: "~것이다"를 버릇처럼 반복하지 말 것. "~의"를 사슬처럼 잇지 말 것
  (예: "당의 노선의 관철의 과정"이 아니라 "당 노선을 관철하는 과정"). "되어지다"
  같은 이중 피동을 쓰지 말 것. он/она를 기계적으로 "그/그녀"로 옮기지 말 것 —
  한국어는 아는 주어를 생략한다. 문맥이 허락하면 한자어+하다보다 고유어 동사를
  고른다.
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
- Корея·корейский는 1948년 이전 문맥에서 "조선"·"조선인"·"조선어"로 옮긴다.
  "한국"은 1948년에 선 남쪽 국가의 이름이므로 그 이전 시기의 문서에는 쓰지 않는다.

출력 형식 (엄격)
- 입력의 각 단락은 [[번호|태그]] 마커로 시작한다. 같은 마커를 같은 순서로 그대로
  반환하고, 마커 바로 다음 줄부터 그 단락의 번역문을 쓴다.
- 마커를 빠뜨리거나, 없는 마커를 만들거나, 두 단락을 한 마커로 합치지 않는다.
- 한 마커 안의 줄바꿈 개수는 원문과 같게 유지한다.
- 마커 줄과 번역문 외에 어떤 텍스트도 출력하지 않는다.

표기 규칙 보강 (위반이 잦아 명시함 — 엄격히 지킬 것)
- 인명의 이니셜(М. С., Ю. Д. 등)은 키릴 문자 그대로 유지한다. 로마자(M. S., Yu. D.)로
  바꾸는 것은 금지.
- 괄호 원어 병기는 위에서 허용한 두 경우(용어표에 없는 인명의 첫 등장, 기관 약어의
  첫 등장)에만 쓴다. 그 밖의 어떤 낱말에도 괄호 병기를 덧붙이지 않는다 — 지명(예레반,
  카라바흐 등), 일반 명사, 이미 나온 이름, 용어표에 있는 항목에 병기를 붙이는 것은 금지.
- 원문에 없는 비칭·경멸 어감을 더하지 않는다 (турок은 "투르크인"이지 "튀르크놈"이 아니다)."""


SYSTEM_PROMPT_ZH = """당신은 1960년대 중국공산당의 공식 문건을 한국어로 옮기는 사료
번역자다. 원문은 《인민일보》·《홍기》 편집부 명의로 발표된 논쟁문이며, 역사 연구용
참고 문헌으로 공개된다.

번역 원칙
- 1차 사료다. 내용을 요약·생략·완곡화·현대화하지 않는다. 원문에 있는 정보는 하나도
  빠뜨리지 않는다. 논지가 거칠거나 상대를 매도하는 대목도 그대로 옮긴다.
- 그러나 중국어 어순을 한국어에 옮기지 않는다. 병렬 구조를 길게 잇는 정론문의 문장을
  그대로 한 문장에 담으려 하면 한국어로 읽을 수 없는 문장이 된다. 필요하면 문장을
  나누고 어순을 한국어에 맞게 재배열한다. 정보 보존이 기준이고, 문장 경계 보존은
  기준이 아니다.
- 한자어를 한국 한자음으로 그대로 읽어 옮기지 않는다. 한국어에서 쓰지 않는 낱말이면
  뜻을 옮긴다. 예: 修正主义는 "수정주의"(한국어에서 쓰는 말이므로 그대로), 그러나
  掩盖는 "엄개"가 아니라 "가리다", 大肆는 "대사"가 아니라 "마구"다.
- 성어와 비유는 뜻을 옮긴다. 直译하면 한국어 독자에게 아무 뜻도 전달되지 않는다.
  예: 掩耳盗铃은 "귀를 막고 방울을 훔친다"로 옮기되 뜻이 통하게 문맥을 살린다.
- 번역투 금지: "~것이다"를 버릇처럼 반복하지 말 것. "~의"를 사슬처럼 잇지 말 것.
  "되어지다" 같은 이중 피동을 쓰지 말 것. 他/她를 기계적으로 "그/그녀"로 옮기지
  말 것 — 한국어는 아는 주어를 생략한다. 문맥이 허락하면 한자어+하다보다
  고유어 동사를 고른다.
- 정치 용어는 아래 용어표를 따른다. 표에 있는 항목은 반드시 그 표기를 쓴다.
- 인용문, 조항 번호, 날짜, 수량, 직위, 문헌 제목은 정확히 보존한다. 마르크스·레닌·
  스탈린 인용은 인용문임이 드러나게 옮긴다.
- 원문에 없는 설명·주석·머리말·꼬리말을 절대 덧붙이지 않는다. 번역문만 출력한다.

표기
- 중국 인명은 중국어 발음으로 음차한다. 마오쩌둥, 저우언라이, 덩샤오핑. 한국 한자음
  (모택동, 주은래)을 쓰지 않는다.
- 러시아어·유럽 인명은 중국어 음역을 되돌려 원어 발음의 한국어 표기를 쓴다.
  赫鲁晓夫는 "허루샤오푸"가 아니라 "흐루쇼프", 斯大林은 "스다린"이 아니라 "스탈린",
  铁托는 "톄퉈"가 아니라 "티토"다. 지명도 같다. 布加勒斯特는 "부쿠레슈티"다.
- 朝鲜은 1948년 이전 문맥에서 "조선", 1948년 이후 조선민주주의인민공화국을 가리키면
  "조선"이다. 韩国은 남쪽 국가를 가리킬 때만 "한국"이며, 1948년 이전 시기에는 쓰지
  않는다.
- 용어표에 없는 중국 인명·지명은 처음 나올 때만 괄호에 한자를 병기한다.
  예: 캉성(康生). 이후에는 한국어 표기만 쓴다. 러시아어·유럽 인명에는 한자를
  병기하지 않는다 — 한자는 중국어 음역일 뿐 그 사람의 이름이 아니다.
- 신문·잡지·단행본 이름은 겹화살괄호를 그대로 쓴다: 《인민일보》, 《홍기》.
  사설·논문·문건·성명의 이름은 홑낫표로 옮긴다: 「소련공산당 중앙위원회 공개서한」.
- 인용한 말과 글에는 굽은 따옴표(“…”)를 쓴다. 홑낫표는 문헌·문건의 이름에만 쓴다.
- 원문이 강조로 쓴 착점(着重号)은 옮기지 않는다. 본문 글자만 옮긴다.

출력 형식 (엄격)
- 입력의 각 단락은 [[번호|태그]] 마커로 시작한다. 같은 마커를 같은 순서로 그대로
  반환하고, 마커 바로 다음 줄부터 그 단락의 번역문을 쓴다.
- 마커를 빠뜨리거나, 없는 마커를 만들거나, 두 단락을 한 마커로 합치지 않는다.
- 한 마커 안의 줄바꿈 개수는 원문과 같게 유지한다.
- 마커 줄과 번역문 외에 어떤 텍스트도 출력하지 않는다."""


class SpecError(ValueError):
    """Spec is missing, malformed, or no longer matches its source."""


@dataclass(frozen=True)
class SourceLanguage:
    """What differs when the source is not Russian.

    The pipeline was built against one corpus and hardcoded its assumptions:
    Cyrillic is what an untranslated word looks like, a Korean rendering is
    about half the length of its source, a glossary surface needs a
    letter-boundary guard. None of that holds for Chinese, where Korean runs
    *longer* than the source and there are no word boundaries to anchor on.
    Rather than branch on the language at each site, the differences are
    gathered here and the spec names one with ``sourceLang``.
    """

    code: str
    label: str
    system_prompt: str
    # What untranslated source text looks like in the output.
    script: re.Pattern
    # A stray run of source script, for the whole-document sweep.
    stray_word: re.Pattern
    # Runs shorter than this are noise rather than a hole (Russian initials
    # such as С.О.). Chinese has no such case: one stray 한자 is one hole.
    stray_min: int
    # Floor on translated/source length before a reply counts as a stub.
    # Russian compresses into Korean, Chinese expands into it.
    short_ratio: float
    # Glossary surfaces get a letter-boundary guard only where the script has
    # word boundaries. Anchoring 毛泽东 on "not a Han character" would refuse
    # to match it in 毛泽东同志, which is most of its occurrences.
    bounded: bool
    # Source chars per input token, and translated chars per source char, for
    # the cost estimate.
    chars_per_token: float
    output_ratio: float
    # Registry feature for this pair (config/llm_call_sites.json). 언어쌍마다
    # 다른 provider·model을 태울 수 있는 라우팅 지점이다. 캐시 키는 feature
    # 이름이 아니라 resolve된 provider·model·thinking으로 만들어지므로,
    # feature를 나눠도 항목 값이 같으면 기존 캐시는 그대로 유효하다.
    feature: str
    # 라틴 문자 저본: 용어표 표면을 인물 familyName.en / 용어 term.en에서 뽑고,
    # 경계 검사와 잔여 검사를 로마자 기준으로 한다.
    latin: bool = False


RUSSIAN = SourceLanguage(
    code="ru", label="러시아어", system_prompt=SYSTEM_PROMPT,
    script=CYRILLIC_RE,
    stray_word=re.compile(r"[А-Яа-яЁёІіЇїЄє][А-Яа-яЁёІіЇїЄє\-]*"),
    stray_min=3, short_ratio=0.25, bounded=True,
    chars_per_token=2.2, output_ratio=0.9,
    feature="archival_document_translation_ru",
)

CHINESE = SourceLanguage(
    code="zh", label="중국어", system_prompt=SYSTEM_PROMPT_ZH,
    script=HAN_RE,
    stray_word=re.compile(r"[㐀-䶿一-鿿]+"),
    stray_min=1, short_ratio=0.7, bounded=False,
    chars_per_token=1.4, output_ratio=1.8,
    feature="archival_document_translation_zh",
)


def _latin_prompt(label: str, era: str, examples: str) -> str:
    """라틴 문자 저본(영어·독일어·프랑스어·이탈리아어) 공용 시스템 프롬프트.
    러시아어 프롬프트와 원칙은 같고, 표기 규칙만 다르다: 인명은 원어 발음대로
    음차하고 첫 등장에서 괄호에 로마자 원문을 그대로 두며, 기관 약어(NATO,
    SHAEF, OKW)는 통용 약어를 그대로 쓴다."""
    return f"""당신은 {era} {label} 공문서를 한국어로 옮기는 사료 번역자다.
원문은 국가·군·정당 기관의 공식 문서(지령, 조약, 외교 전문, 회의록, 성명)이며, 역사
연구용 참고 문헌으로 공개된다.

번역 원칙
- 1차 사료다. 내용을 요약·생략·완곡화·현대화하지 않는다. 원문에 있는 정보는 하나도
  빠뜨리지 않는다.
- 그러나 {label} 어순을 한국어에 옮기지 않는다. 긴 복문, 삽입절, 명사구 연쇄를
  그대로 한 문장에 담으려 하면 한국어로 읽을 수 없는 문장이 된다. 필요하면 문장을
  나누고 어순을 한국어에 맞게 재배열한다. 정보 보존이 기준이고, 문장 경계 보존은
  기준이 아니다.
- 완성된 번역문은 한국어 문어로 자연스럽게 읽혀야 한다. 번역투로 뻣뻣하더라도
  원문 구조를 흉내 내는 쪽을 택하지 말 것.
- 번역투 금지: "~것이다"를 버릇처럼 반복하지 말 것. "~의"를 사슬처럼 잇지 말 것.
  "되어지다" 같은 이중 피동을 쓰지 말 것. 대명사를 기계적으로 "그/그녀/그것"으로
  옮기지 말 것 — 한국어는 아는 주어를 생략한다. 문맥이 허락하면 한자어+하다보다
  고유어 동사를 고른다.
- 원문의 관료적·군사적 문체와 완곡어법 자체는 그대로 옮긴다. 원문이 모호하면 모호한
  채로 둔다. 다의어를 사전 첫 번째 뜻으로 기계적으로 옮기지 말 것. {examples}
- 문서 번호, 날짜, 조항 번호, 수량, 직위, 서명, 문서보관소 출처 표기는 정확히 보존한다.
- 원문에 없는 설명·주석·머리말·꼬리말을 절대 덧붙이지 않는다. 번역문만 출력한다.

표기
- 기관·직위·용어는 아래 용어표를 따른다. 용어표에 있는 항목은 반드시 그 표기를 쓴다.
- 용어표에 없는 인명은 원어 발음에 따라 음차하고, 처음 나올 때만 괄호에 로마자 원문을
  그대로 병기한다. 이후에는 한국어 표기만 쓴다.
- 기관·작전·무기 약어(NATO, SHAEF, OKW, ICBM 등)는 용어표 표기를 쓰되, 용어표에 없으면
  처음 나올 때 풀어 쓰고 괄호에 통용 약어를 둔 뒤 이후에는 약어만 쓴다. 로마자 약어는
  번역문에 그대로 남겨도 된다.
- 지명은 통용 한국어 표기(런던, 베를린, 바르샤바)를 쓰고, 통용 표기가 없는 작은 지명은
  원어 발음대로 음차하며 처음 나올 때 괄호에 원문을 둔다.
- Korea·Korean은 1948년 이전 문맥에서 "조선"·"조선인"으로 옮긴다.

출력 형식 (엄격)
- 입력의 각 단락은 [[번호|태그]] 마커로 시작한다. 같은 마커를 같은 순서로 그대로
  반환하고, 마커 바로 다음 줄부터 그 단락의 번역문을 쓴다.
- 마커를 빠뜨리거나, 없는 마커를 만들거나, 두 단락을 한 마커로 합치지 않는다.
- 한 마커 안의 줄바꿈 개수는 원문과 같게 유지한다.
- 마커 줄과 번역문 외에 어떤 텍스트도 출력하지 않는다."""


# 라틴 문자 저본의 「미번역 잔여」: 번역문에는 약어(NATO, ICBM)와 첫 등장의 괄호 원문이
# 정당하게 남으므로, 둘째 글자가 소문자인 4자 이상 낱말만 미번역으로 본다. 괄호 안은
# stray_cyrillic이 이미 빼고 본다.
_LATIN_STRAY = re.compile(r"\b[A-Za-zÀ-ÖØ-öø-ÿ][a-zà-öø-ÿß][A-Za-zÀ-ÖØ-öø-ÿ'’\-]+")


def _latin(code: str, label: str, era: str, examples: str, feature: str,
           chars_per_token: float, output_ratio: float) -> "SourceLanguage":
    return SourceLanguage(
        code=code, label=label, system_prompt=_latin_prompt(label, era, examples),
        script=LATIN_RE, stray_word=_LATIN_STRAY,
        stray_min=4, short_ratio=0.2, bounded=True,
        chars_per_token=chars_per_token, output_ratio=output_ratio,
        feature=feature, latin=True,
    )


ENGLISH = _latin("en", "영어", "20세기",
                 "예: \"appreciation\"은 군사 문서에서 \"감사\"가 아니라 \"정세 판단\"이다.",
                 "archival_document_translation_en", 4.0, 0.55)
GERMAN = _latin("de", "독일어", "20세기",
                "예: \"Weisung\"은 \"지시\"가 아니라 총통 \"지령\"이고, \"Aufmarsch\"는 \"행진\"이 아니라 \"전개(집결)\"다.",
                "archival_document_translation_de", 3.5, 0.5)
FRENCH = _latin("fr", "프랑스어", "20세기",
                "예: \"ordonnance\"는 해방기 임시정부 문맥에서 \"명령\"(법률 효력의 명령)이다.",
                "archival_document_translation_fr", 3.8, 0.5)
ITALIAN = _latin("it", "이탈리아어", "20세기",
                 "예: \"ordine del giorno\"는 \"일정\"이 아니라 회의에 제출된 \"결의안\"이다.",
                 "archival_document_translation_it", 3.8, 0.5)

LANGUAGES = {lang.code: lang for lang in (RUSSIAN, CHINESE, ENGLISH, GERMAN, FRENCH, ITALIAN)}


def language_for(spec: dict | None) -> SourceLanguage:
    """The spec's source language. Specs written before this existed have no
    field and are Russian, which is what they were parsed and priced as."""
    code = ((spec or {}).get("sourceLang") or "ru").strip()
    lang = LANGUAGES.get(code)
    if lang is None:
        raise SpecError(
            f"알 수 없는 sourceLang: {code!r} (쓸 수 있는 값: "
            f"{', '.join(sorted(LANGUAGES))})")
    return lang


@dataclass
class Options:
    """실행 옵션. 모델·max_tokens·thinking은 여기 없다 — 언어쌍별 registry
    항목(config/llm_call_sites.json)이 결정하고, 호출부에서 넘긴 값은
    registry가 무시한다. 바꾸려면 그 파일이나 LLM_SITE_*_MODEL 환경변수."""
    max_chars: int = 3500
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
    # 캐시에 있었지만 현재 검증기를 통과하지 못해 다시 번역한 청크 수.
    revalidated: int = 0
    # 워커 스레드들이 같은 객체를 올린다. 속성 += 는 원자적이지 않으므로
    # 갱신은 add()로 모은다.
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def add(self, name: str, n: int = 1) -> None:
        with self._lock:
            setattr(self, name, getattr(self, name) + n)

    def as_dict(self) -> dict:
        return {"cached": self.cached, "translated": self.translated,
                "retried": self.retried, "failed": self.failed,
                "revalidated": self.revalidated}


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
    return sources.parse(source, _decode_page(src.read_bytes()))


_CHARSET_RE = re.compile(rb'charset=["\']?\s*([A-Za-z0-9_-]+)', re.I)


def _decode_page(data: bytes) -> str:
    """저장된 페이지 바이트 → 문자열. 1990년대 문서 사이트(관보 전자판 등)는
    windows-1251이라 UTF-8로 읽으면 본문이 통째로 깨진다. 선언된 charset을
    따르고, 없으면 UTF-8을 시도한 뒤 cp1251로 물러선다."""
    m = _CHARSET_RE.search(data[:4096])
    declared = (m.group(1).decode("ascii", "ignore").lower() if m else "")
    if declared in ("windows-1251", "cp1251", "win-1251"):
        return data.decode("cp1251", errors="replace")
    # 라틴 문자 저본(documentarchiv.de ISO-8859-1, der-fuehrer.org cp1252)도
    # 선언을 따른다 — UTF-8 실패 뒤 cp1251로 물러서면 움라우트가 키릴로 깨진다.
    if declared and declared not in ("utf-8", "utf8"):
        try:
            return data.decode(declared, errors="replace")
        except LookupError:
            pass
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("cp1251", errors="replace")


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
        # tmExamples(스펙 또는 문서 항목에 고정한 확정 번역례)도 같은 길로
        # 나른다 — 실행 시점에 TM에서 동적으로 뽑으면 TM이 자랄 때마다
        # 프롬프트와 캐시 키가 바뀌어, postEdits 수정 하나에 문서 전체를
        # 재번역하게 된다. 후보 추천은 scripts/suggest_tm_examples.py가 하고,
        # 채택은 사람이 스펙을 고쳐서 한다(의도된 캐시 무효화).
        examples = entry.get("tmExamples") or spec.get("tmExamples")
        chosen = [{**b, "register": entry.get("register"), "tmExamples": examples}
                  for b in chosen]
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


def _pattern(surfaces: list[str], bounded: bool = True) -> re.Pattern:
    alts = "|".join(sorted((re.escape(s) for s in surfaces), key=len, reverse=True))
    if not bounded:
        return re.compile(alts)
    # 대안은 반드시 묶는다. 묶지 않으면 lookbehind는 첫 대안에만, lookahead는 마지막
    # 대안에만 걸려 "Горбаче"(Горбач의 전치격)가 "Горбачев" 안에서 매칭된다 —
    # 1989 결의 감사에서 고르바초프를 그리고리 고르바치로 바꾸라는 제안이 나온 원인.
    cls = "А-Яа-яЁё" if any(CYRILLIC_RE.search(x) for x in surfaces) else "A-Za-zÀ-ÖØ-öø-ÿ"
    return re.compile(rf"(?<![{cls}])(?:{alts})(?![{cls}])")


def build_glossary(people_path: Path, terms_path: Path,
                   extra: dict[str, str] | None = None,
                   lang: SourceLanguage | None = None) -> list[dict]:
    """[{ru, ko, pattern}] — pattern is what a chunk is searched for.

    ``extra`` is added first so it wins: the people/terms dictionaries are
    keyed to concepts and people, and carry no entry for the bare institution
    abbreviations (НКВД, ГУГБ, ЦК ВКП(б)) that saturate these documents. Left
    unpinned the model invents a rendering per chunk — it produced
    "인민내무위원부" for НКВД, a word order this site never uses.
    """
    lang = lang or RUSSIAN
    seen: set[str] = set()
    entries: list[dict] = []

    def add(display_ru: str, ko: str, surfaces: list[str]) -> None:
        if display_ru in seen or not ko:
            return
        seen.add(display_ru)
        entries.append({"ru": display_ru, "ko": ko,
                        "pattern": _pattern(surfaces, lang.bounded)})

    for ru, ko in (extra or {}).items():
        add(ru, ko, [ru])

    people = json.loads(people_path.read_text(encoding="utf-8")).get("people", [])
    for p in people:
        if lang.latin:
            # 라틴 문자 저본: 사전의 영어 성(姓)이 표면이다. 독일어·프랑스어 문서의
            # 러시아·독일 인명은 영어 표기와 대체로 같고, 다르면 스펙 extra로 준다.
            fam_en = ((p.get("familyName") or {}).get("en") or "").strip()
            fam_ko = ((p.get("familyName") or {}).get("ko") or "").strip()
            if len(fam_en) >= 4 and fam_ko and LATIN_RE.search(fam_en):
                add(fam_en, fam_ko, [fam_en])
            continue
        cyr = (p.get("cyrillic") or "").strip()
        if not cyr or not lang.script.search(cyr):
            continue
        if lang.bounded:
            family_ko = (p.get("familyName") or {}).get("ko")
            if not family_ko:
                continue
            # Family name only. A given name on its own is too common to pin
            # to one person by surface match. The length floor is 4, not 5:
            # Ежов is four letters, and boundary+case-ending matching (not
            # substring) already carries the disambiguation a length filter
            # used to.
            family_ru = cyr.split()[-1]
            if len(family_ru) >= 4:
                add(family_ru, family_ko, _variants(family_ru))
            continue
        # Chinese: pin the whole name. A Chinese surname is one character and
        # matching it alone would fire on every 李 in the text; the full name
        # is what the source actually writes, and it is unambiguous.
        full_ko = ((p.get("name") or {}).get("ko") or "").strip()
        if len(cyr) >= 2 and full_ko:
            add(cyr, full_ko, [cyr])

    for t in json.loads(terms_path.read_text(encoding="utf-8")):
        original = (t.get("original") or "").strip()
        term = ((t.get("term") or {}).get("ko") or "").strip()
        if lang.latin:
            # 영어 표제어 가운데 고유명사꼴(대문자로 시작, 5자 이상)만 표면으로
            # 쓴다. "party", "terror" 같은 보통명사를 사전 항목에 묶으면 문맥마다
            # 틀린 표기를 강요하게 된다. original이 로마자면 그것도 표면에 더한다.
            en = ((t.get("term") or {}).get("en") or "").strip()
            surfaces = [x for x in (en, original)
                        if x and len(x) >= 5 and x[0].isupper() and LATIN_RE.search(x)
                        and not CYRILLIC_RE.search(x)]
            if surfaces and term:
                add(surfaces[0], term, surfaces)
            continue
        if original and term and lang.script.search(original):
            add(original, term, [original])

    return entries


def glossary_entries_for(text: str, glossary: list[dict], limit: int) -> list[dict]:
    """이 텍스트에 등장하는 용어표 항목들 (프롬프트에 참고로 주입된다)."""
    hits = [g for g in glossary if g["pattern"].search(text)]
    return hits[:limit]


def glossary_for(text: str, glossary: list[dict], limit: int) -> list[tuple[str, str]]:
    return [(g["ru"], g["ko"]) for g in glossary_entries_for(text, glossary, limit)]


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


def validate(chunk: list[tuple[int, dict]], got: dict[int, list[str]],
             lang: SourceLanguage | None = None) -> list[str]:
    """형식·누락·미번역·길이만 본다. 용어표 준수는 검사하지 않는다 — 표면
    일치로는 다의어(Союз, Правда, Октябрьский)와 인물 격변화 충돌을 가릴 수
    없고, 그 검사가 교정 재시도로 모델의 옳은 첫 번역을 뒤집어 놓은 적이
    있다(1925 대회 문서). 용어표는 프롬프트에 참고로 주입될 뿐이고, 표기
    통일은 발행 전 통독과 스펙 postEdits가 맡는다."""
    lang = lang or RUSSIAN
    problems = []
    # A block with no lines (an all-numeric table) sends only its marker and
    # owes no reply — do not fail the chunk when the model skips it.
    expected = {idx for idx, b in chunk if b["lines"]}
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
        src_cyr = len(lang.script.findall(source))

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
        cyr = len(lang.script.findall(outside))
        if cyr / max(len(outside.strip()), 1) > 0.15:
            problems.append(
                f"[[{idx}]] {lang.label}가 그대로 남음 ({cyr}자): {outside.strip()[:40]}…")

        # Korean renders Russian in roughly half the characters, but a sentence
        # dense in long compound nouns (мобилизационная подготовка → 동원 준비)
        # compresses much further. Measured over 170 blocks of this corpus:
        # median 0.50, minimum 0.34. A 0.35 floor sat inside the real
        # distribution and failed a correct translation; 0.25 clears every
        # observed block while still catching a stub reply to a long paragraph.
        # (Chinese runs the other way — one 한자 becomes two or three 한글 — so
        # its floor sits above 1.0 rather than below it.)
        src_len = sum(len(ln) for ln in block["lines"])
        if src_len > 200 and len(joined) < src_len * lang.short_ratio:
            problems.append(f"[[{idx}]] 번역문이 지나치게 짧음 ({len(joined)}자 < 원문 {src_len}자)")

    # 응답이 도중에 끊긴 청크. 끊김은 청크의 마지막 블록에서만 조용히 지나간다 —
    # 그 앞에서 끊기면 뒤 블록의 마커가 빠져 위에서 잡히고, 마지막 블록은
    # 길이 하한만 넘으면 통과했다. 1925 대회 문서의 블록 43·1000017이 그렇게
    # "…만들어졌", "…지원"에서 끊긴 채(원문 대비 0.36·0.30) 발행됐다. 원문이
    # 문장부호로 끝나는데 번역의 마지막 줄이 그렇지 않으면 끊긴 것으로 본다.
    # 전 스펙 캐시 2,923블록 실측: 마지막 블록 제한 아래 실제 끊김 2건만 잡고
    # 오탐 0건 (제한 없이는 제목·목록 항목 3건이 걸린다). 청크 끝에 제목이 오는
    # 일은 chunk_document가 막으므로 태그 예외는 필요 없다.
    last = next((idx for idx, b in reversed(chunk) if b["lines"]), None)
    if last is not None and got.get(last):
        source = " ".join(dict(chunk)[last]["lines"]).strip()
        tail = got[last][-1].strip()
        if (len(source) >= 80 and _SENTENCE_END_SRC.search(source)
                and not _SENTENCE_END_TGT.search(tail)):
            problems.append(f"[[{last}]] 응답이 도중에 끊김: …{tail[-30:]}")
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

def _prepare_chunk(chunk, glossary, opts: Options,
                   lang: SourceLanguage | None = None) -> tuple[str, str]:
    """(prompt, cache key) for one chunk — run()의 pending 집계와 워커가 같은
    키를 봐야 하므로 한 자리에서 한 번만 만든다."""
    body = render_chunk(chunk)
    terms = glossary_for(body, glossary, opts.glossary_limit)
    prompt = _render_prompt(chunk, body, terms)
    return prompt, _chunk_key(prompt, opts, lang)


def _chunk_prompt(chunk, glossary, opts: Options) -> str:
    return _prepare_chunk(chunk, glossary, opts)[0]


def _render_prompt(chunk, body: str, terms: list[tuple[str, str]]) -> str:
    gloss_text = "\n".join(f"- {ru} → {ko}" for ru, ko in terms) or "(해당 없음)"
    example_text = ""
    examples = chunk[0][1].get("tmExamples") or []
    if examples:
        rendered = "\n".join(
            f"- 원문: {e['source']}\n  번역: {e['target']}" for e in examples)
        example_text = ("참고 번역례 (이 서고의 확정 번역 — 표기와 문체를 따르되 "
                        f"문장을 그대로 옮겨 쓰지 말 것)\n{rendered}\n\n")
    register = (chunk[0][1].get("register") or "").strip()
    register_line = f"문체: {register}\n\n" if register else ""
    return (f"용어표 (반드시 이 표기를 쓸 것)\n{gloss_text}\n\n{example_text}{register_line}"
            f"아래 단락들을 번역하라.\n\n{body}")


def _chunk_key(prompt: str, opts: Options,
               lang: SourceLanguage | None = None) -> str:
    """Cache key. Resolving the profile needs no credential, so this is safe
    to compute before preflight.

    The provider and its thinking setting are part of the key, not just the
    model: a chunk translated over the OpenAI-compatible path with thinking
    on is a different artifact from the same chunk over the Anthropic-
    compatible path with it off, and reusing one for the other silently
    carries stale output across a config change.
    """
    from llm import call_registry

    lang = lang or RUSSIAN
    profile = call_registry.resolve(lang.feature)
    # The system prompt belongs in the key too. Without it, editing the
    # translation rules silently reuses output produced under the old ones
    # unless someone remembers to bump PROMPT_VERSION by hand — and a
    # constant that has to be remembered is a constant that gets forgotten.
    system_hash = hashlib.sha256(
        lang.system_prompt.encode("utf-8")).hexdigest()[:16]
    fingerprint = (f"{profile.provider}\0{profile.model}\0"
                   f"{profile.extra.get('thinking')}\0{system_hash}")
    return hashlib.sha256(
        f"{PROMPT_VERSION}\0{fingerprint}\0{prompt}".encode("utf-8")).hexdigest()


def _cached_blocks(cache: "Cache", key: str, chunk,
                   lang: SourceLanguage | None = None) -> tuple[dict[int, list[str]] | None, list[str]]:
    """캐시 레코드를 현재 검증기로 다시 심사해 (블록들, 문제들)을 돌려준다.

    캐시는 그 레코드를 저장할 당시의 validate()를 통과했다는 뜻이지 지금의
    validate()를 통과했다는 뜻이 아니다. 검사가 하나 늘었을 때 PROMPT_VERSION을
    올려 문서 전체를 다시 번역하는 대신, 새 검사에 걸리는 청크만 miss로 취급해
    그것만 다시 번역한다. 문제가 없으면 (블록들, []), 레코드가 없으면 (None, []),
    있지만 걸리면 (None, 문제들)."""
    rec = cache.get(key)
    if rec is None:
        return None, []
    blocks = {int(k): v for k, v in rec["blocks"].items()}
    problems = validate(chunk, blocks, lang)
    return (None, problems) if problems else (blocks, [])


def _translate_chunk(chunk, glossary, cache, opts: Options, stats: Stats,
                     progress: Callable[[dict], None],
                     lang: SourceLanguage | None = None,
                     prepared: tuple[str, str] | None = None) -> dict[int, list[str]]:
    from llm import call_registry

    lang = lang or RUSSIAN
    # run()은 pending 집계 때 이미 만든 것을 넘긴다; 직접 부르는 쪽은 여기서 만든다.
    prompt, key = prepared or _prepare_chunk(chunk, glossary, opts, lang)

    cached, stale = _cached_blocks(cache, key, chunk, lang)
    if cached is not None:
        stats.add("cached")
        return cached
    if stale:
        stats.add("revalidated")
        progress({"event": "cacheInvalid", "blocks": [chunk[0][0], chunk[-1][0]],
                  "problems": stale[:3]})

    correction = ""
    last_reason = "원인 미상"
    for attempt in range(1, opts.retries + 1):
        raw = call_registry.generate_sync(
            lang.feature, prompt + correction, system=lang.system_prompt)
        if not raw:
            # generate_sync swallows the provider exception and returns None,
            # logging the cause itself. Surface at least that it happened —
            # a silent retry here is how a bad key burns every chunk before
            # anyone sees why.
            last_reason = "빈 응답 (provider 오류는 llm-registry 로그 참조)"
            stats.add("retried")
            progress({"event": "retry", "blocks": [chunk[0][0], chunk[-1][0]],
                      "attempt": attempt, "problems": [last_reason]})
            correction = "\n\n(직전 응답이 비어 있었다. 형식을 지켜 다시 출력하라.)"
            time.sleep(2 * attempt)
            continue
        got = parse_response(raw)
        problems = validate(chunk, got, lang)
        if not problems:
            cache.put(key, got, {"attempt": attempt, "chars": len(prompt)})
            stats.add("translated")
            return got
        correction = (
            "\n\n(직전 응답에 다음 문제가 있었다. 같은 입력을 형식에 맞게 다시 번역하라.\n"
            + "\n".join(f"- {p}" for p in problems) + ")"
        )
        last_reason = "; ".join(problems[:3])
        stats.add("retried")
        progress({"event": "retry", "blocks": [chunk[0][0], chunk[-1][0]],
                  "attempt": attempt, "problems": problems[:3]})

    stats.add("failed")
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
        if not re.search(r"[А-Яа-яЁёІіЇїЄєA-Za-z㐀-䶿一-鿿]", inner):
            return m.group(0)  # Korean parenthetical: part of the name
        if korean in seen:
            return korean
        seen.add(korean)
        return m.group(0)

    return lambda line: _GLOSS_RE.sub(repl, line)


def dedupe_glosses(lines: list[str]) -> list[str]:
    apply = gloss_deduper()
    return [apply(line) for line in lines]


def stray_cyrillic(text: str, allowed: list[str] | None = None,
                   lang: SourceLanguage | None = None) -> list[str]:
    """Source-script words left outside parentheses that may not remain.

    A single untranslated word inside a long paragraph is invisible to the
    per-block ratio check, so it needs a whole-document pass. Document code
    names the translation deliberately keeps (ПОВ, КН-1) go in the spec's
    allowedCyrillic list; anything else is a hole in the translation.
    """
    lang = lang or RUSSIAN
    # 엮은이 주 상자(해제·서지)는 엮은이가 쓴 글이지 번역 출력이 아니다. 서지의
    # 원제(«Приказ НКО СССР № 227…»)가 통째로 미번역 잔여로 찍혀, 스펙마다
    # 그 낱말들을 allowedCyrillic에 베껴 넣어야 했다. 상자는 검사에서 뺀다.
    text = re.sub(r"(?is)<aside class=\"doc-editorial\">.*?</aside>", " ", text)
    outside = re.sub(r"[(（][^)）]*[)）]", " ", re.sub(r"<[^>]+>", " ", text))
    keep = set(allowed or [])
    found = []
    for word in lang.stray_word.findall(outside):
        if word in keep or any(word.startswith(k) for k in keep):
            continue
        if len(word) < lang.stray_min:  # Russian initials such as С.О.
            continue
        found.append(word)
    return sorted(set(found))


# 앞말의 받침에 따라 모양이 바뀌는 조사. (모음 뒤 꼴, 받침 뒤 꼴). 긴 것이
# 먼저 오도록 나열한다 — "이라"를 "이"로 먼저 잡으면 안 된다.
_PARTICLE_PAIRS = [
    ("였", "이었"), ("라", "이라"), ("란", "이란"), ("나", "이나"), ("든", "이든"),
    ("며", "이며"), ("랑", "이랑"), ("여", "이여"), ("로", "으로"),
    ("는", "은"), ("가", "이"), ("를", "을"), ("와", "과"), ("야", "아"),
]
_AFTER_VOWEL = {v: c for v, c in _PARTICLE_PAIRS}
_AFTER_CODA = {c: v for v, c in _PARTICLE_PAIRS}


def _coda(ch: str) -> int | None:
    """한글 음절의 받침 번호 (0=없음, 8=ㄹ), 한글 음절이 아니면 None."""
    o = ord(ch)
    return (o - 0xAC00) % 28 if 0xAC00 <= o <= 0xD7A3 else None


def _repair_particle(dst: str, tail: str) -> str:
    """치환한 낱말 ``dst`` 바로 뒤에 오는 조사 ``tail``이 dst의 받침과 어긋나면 고친다.

    문자열 치환은 앞말만 바꾸고 조사는 그대로 두므로 "소비에트 소유즈와/로/는"이
    "소비에트 연방와/로/는"이 되어 발행됐다(1925 대회 문서). 판정은 치환 자리
    바로 뒤, 그것도 dst의 받침과 **어긋나는** 조사에만 한다 — 맞는 조사와
    조사가 아닌 글자는 건드리지 않는다. 받침 뒤 꼴을 모음 뒤 꼴로 되돌리는
    방향은 은/이/을/과/으로만 한다: "이다/이라/이었"은 모음 뒤에서도 문어에서
    허용되는 꼴이라 틀렸다고 볼 수 없다. ㄹ받침 뒤의 "로"는 옳으므로 그대로,
    "으로"는 "로"로 줄인다.
    """
    coda = _coda(dst[-1]) if dst else None
    if coda is None:
        return tail
    if coda == 0:
        # 모음 뒤에서도 허용되는 '이-' 계열은 먼저 걸러 '이'로 오인하지 않는다.
        if tail.startswith(("이었", "이라", "이란", "이나", "이든", "이며", "이랑", "이여", "이다")):
            return tail
        for c, v in (("으로", "로"), ("은", "는"), ("이", "가"), ("을", "를"), ("과", "와")):
            if tail.startswith(c):
                return v + tail[len(c):]
        return tail
    for v, c in _PARTICLE_PAIRS:
        if tail.startswith(v):
            if v == "로" and coda == 8:
                return tail
            return c + tail[len(v):]
    if coda == 8 and tail.startswith("으로"):
        return "로" + tail[len("으로"):]
    return tail


def _replace_with_particles(line: str, src: str, dst: str) -> str:
    if src not in line:
        return line
    out, pos = [], 0
    while True:
        i = line.find(src, pos)
        if i < 0:
            out.append(line[pos:])
            return "".join(out)
        out.append(line[pos:i])
        out.append(dst)
        rest = line[i + len(src):]
        fixed = _repair_particle(dst, rest[:3])
        if fixed != rest[:3]:
            out.append(fixed)
            pos = i + len(src) + 3
        else:
            pos = i + len(src)


def apply_post_edits(lines: list[str], spec: dict) -> list[str]:
    """spec.postEdits 치환을 블록 줄들에 적용한다. 치환 뒤 조사도 맞춘다.

    조립기의 fix()가 하는 세 가지 중 postEdits만 가져온다. 따옴표 정규화는
    표기 취향이고, 주석 병기 축약(gloss dedupe)은 문서 안의 위치(첫 등장)에
    묶여 있어 세그먼트 단위로 옮기면 틀린다. postEdits는 오역 교정이므로
    세그먼트 자체의 품질이고, TM에 적재되는 쌍에도 반영되어야 발행본과 같은
    텍스트가 남는다. 조립기도 같은 함수를 쓴다 — 두 군데가 따로 치환하면
    조사 처리처럼 한쪽만 고쳐지는 일이 생긴다.
    """
    edits = spec.get("postEdits") or {}
    if not edits:
        return lines
    out = []
    for line in lines:
        for src, dst in edits.items():
            line = _replace_with_particles(line, src, dst)
        out.append(line)
    return out


def assemble(spec: dict, docs: list[dict], translated: dict[int, list[str]]) -> str:
    # Mechanical repairs for words the model left in Russian mid-sentence.
    # They live in the spec, not in the fragment, so a re-run from cache
    # reproduces the fix instead of silently dropping it.
    edits = spec.get("postEdits") or {}
    dedupe = gloss_deduper()

    # 서고의 관행은 인용문에 따옴표, 문헌·사건 이름에 홑낫표다. 모델은 청크마다
    # 곧은 따옴표와 굽은 따옴표를 섞어 내놓으므로 짝이 맞는 것만 굽은 쪽으로
    # 모은다. 인치 기호나 한쪽만 남은 따옴표는 건드리지 않는다.
    curly_quotes = (spec.get("quotes") == "curly")
    _STRAIGHT_PAIR = re.compile(r'"([^"\n]{1,400})"')

    def fix(line: str) -> str:
        if curly_quotes:
            line = _STRAIGHT_PAIR.sub("\u201c\\1\u201d", line)
        line = apply_post_edits([line], spec)[0] if edits else line
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
                # A block with nothing to translate (an all-numeric table has
                # an empty cell vocabulary) owes no reply; its content is
                # emitted from the block itself. Only a block that actually
                # carried source text is a hole.
                if block["lines"]:
                    raise SpecError(f"조립 중 누락된 블록: {idx}")
                lines = []
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
                # Cells carry note refs too — Bukharin cites his source on a
                # table's total row — so they get the same [n] linking as
                # prose.
                body = "".join(
                    "<tr>" + "".join(
                        f"<td>{link_refs(_esc(vocab.get(c, c)))}</td>" for c in row
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

def preflight(opts: Options | None = None, lang: SourceLanguage | None = None,
              provider: str | None = None) -> None:
    """Fail fast on a credential that cannot possibly work.

    call_registry.generate_sync swallows the provider exception and returns
    None, so without this check a bad key surfaces only as every chunk
    failing its full retry budget with no stated cause. 언어쌍마다 provider가
    다를 수 있으므로 검사할 feature도 lang을 따른다. ``provider``를 주면
    registry 대신 그 provider를 검사한다 (--compare의 변형들).
    """
    lang = lang or RUSSIAN
    from llm import call_registry

    provider = provider or call_registry.resolve(lang.feature).provider
    try:
        connection = call_registry.resolve_provider_connection(provider)
    except ValueError:
        return  # unfamiliar provider shape — let the call itself report
    except call_registry.ProviderConnectionError as exc:
        raise SpecError(
            f"{exc.credential_name}가 설정되어 있지 않다 "
            f"(provider={provider}). credstore를 마운트해 실행하거나 "
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
    lang = language_for(spec)
    from llm import call_registry

    profile = call_registry.resolve(lang.feature)
    executor = call_registry._EXECUTORS.get(profile.provider)
    if executor is None:
        raise SpecError(f"등록되지 않은 provider: {profile.provider}")

    sample = ("다음을 한국어로: Приказ народного комиссара."
              if lang.code == "ru" else "다음을 한국어로: 中央委员会的决定。")
    cases: list[tuple[str, str, str]] = [
        ("minimal", "당신은 번역기다.", sample),
    ]
    if spec is not None:
        prepared = plan(spec, Options(**{**opts.__dict__, "limit_chunks": 1}))
        chunk = prepared["_chunks"][0]
        body = render_chunk(chunk)
        terms = glossary_for(body, prepared["_glossary"], opts.glossary_limit)
        gloss = "\n".join(f"- {ru} → {ko}" for ru, ko in terms) or "(해당 없음)"
        cases.append((
            f"first-chunk ({len(body):,}자)", prepared["_lang"].system_prompt,
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
            raw = executor(profile, prompt, system)
            if isinstance(raw, tuple):
                if len(raw) > 2:
                    record["truncated"] = bool(raw[2])
                raw = raw[0]
            content = raw or ""
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
    chunks, glossary, lang = prepared["_chunks"], prepared["_glossary"], prepared["_lang"]
    base = call_registry.resolve(lang.feature)

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
                raw = executor(profile, prompt, lang.system_prompt)
            except Exception as e:
                error = f"{type(e).__name__}: {e}"
                break
            seconds += time.time() - started
            # Executors return (text, usage) or (text, usage, truncated)
            # since the 2026-08-30 truncation guard; older ones returned str.
            if isinstance(raw, tuple):
                if len(raw) > 2 and raw[2]:
                    problems.append(
                        f"청크 {chunk[0][0]}: max_tokens로 잘림 (truncated)")
                raw = raw[0]
            got = parse_response(raw or "")
            problems.extend(validate(chunk, got, lang))
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


def glossary_collision_pairs(glossary: list[dict]) -> list[tuple[str, str]]:
    """(여성형/남성 생격 표면, 남성형) ru 쌍 — 함께 주입되면 위험한 충돌.

    러시아어 남성 성의 생격·대격(-а)은 같은 성의 여성형 주격과 표기가 같다.
    사전에 두 사람이 모두 있으면 남성형의 굴절 표면(доклад Ежова)이 여성
    항목에 매칭되어 잘못된 표기가 주입된다 — 포스펠로프 보고서의
    Ежова→예조바 사례. 주입 내용은 건드리지 않고 운영자에게 경고만 한다;
    여성 본인이 등장하지 않는 문서는 spec glossary.exclude로 뺀다.
    """
    by_ru = {}
    for g in glossary:
        by_ru.setdefault(g["ru"], g["ko"])
    return sorted((ru + "а", ru) for ru, ko in by_ru.items()
                  if by_ru.get(ru + "а") not in (None, ko))


def plan(spec: dict, opts: Options | None = None) -> dict:
    """Slice, chunk and price a run without calling the model."""
    opts = opts or Options()
    lang = language_for(spec)
    docs = slice_documents(spec)
    glossary = build_glossary(Path(spec["glossary"]["people"]),
                              Path(spec["glossary"]["terms"]),
                              spec["glossary"].get("extra"), lang)
    # glossary.exclude: 이 스펙에서는 용어표에서 아예 빼는 항목의 원문 표기
    # (이 문서의 문맥에서 거의 항상 다른 뜻인 다의어). 주입 목록이 바뀌므로
    # 해당 청크의 캐시 키도 바뀐다.
    exclude = set(spec["glossary"].get("exclude") or [])
    if exclude:
        glossary = [g for g in glossary if g["ru"] not in exclude]
    doc_text = " ".join(ln for d in docs for b in d["blocks"] for ln in b["lines"])
    for fem, masc in glossary_collision_pairs(glossary):
        if fem in doc_text:
            print(f"경고: 용어표 충돌 — 원문의 {fem}이 {masc}의 생격 표면일 수 있는데 "
                  f"여성형 항목이 주입된다. 이 문서에 {fem} 본인이 등장하지 않으면 "
                  f"spec glossary.exclude에 {fem!r}를 넣을 것.", file=sys.stderr)
    chunks = [c for d in docs for c in chunk_document(d, opts.max_chars)]
    if opts.limit_chunks:
        chunks = chunks[: opts.limit_chunks]
    total = sum(len(ln) for c in chunks for _, b in c for ln in b["lines"])

    # Price against the model that will actually run, not a hardcoded tier:
    # a stale estimate is worse than none once the call site can change model.
    from llm import call_registry
    from llm.provider_registry import openai_compatible_pricing

    profile = call_registry.resolve(lang.feature)
    price = openai_compatible_pricing(profile.model)
    thinking_on = (profile.extra.get("thinking") or {}).get("type") == "enabled"
    tokens_in = total / lang.chars_per_token
    # Reasoning tokens bill as output; high effort roughly doubles it.
    tokens_out = total * lang.output_ratio / 1.6 * (2.0 if thinking_on else 1.0)
    est = tokens_in * price["input"] + tokens_out * price["output"]
    return {
        "id": spec.get("id"),
        "sourceLang": lang.code,
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
        "_docs": docs, "_glossary": glossary, "_chunks": chunks, "_lang": lang,
    }


# 자동 재사용은 검수 등급만: machine 세그먼트를 되먹이면 미검수 출력이
# 자기 강화 루프를 탄다. 세그먼트 승격은 backfill(frozen→published)이나
# 수동 reviewed 지정으로 이뤄진다.
_TM_REUSE_STATUSES = ("published", "reviewed")


def _tm_prefill(docs: list[dict], lang: SourceLanguage,
                emit: Callable[[dict], None]) -> dict[int, list[str]]:
    """검수 등급 TM 세그먼트와 완전 일치하는 블록을 모델 없이 채운다.

    반복 문구(조문 서두, 서명부, 재수록 단락)는 이미 사람 검수를 거친 번역이
    코퍼스에 있다. 완전 일치는 LLM 호출 비용이 0이고 회귀 위험도 0이다
    (인수인계 §2.3). TM 실패는 재사용을 포기할 이유일 뿐 번역을 멈출 이유가
    아니므로 어떤 예외도 여기서 멈춘다.
    """
    try:
        from runtime_tools import translation_memory

        source_by_idx: dict[int, str] = {}
        for doc in docs:
            for i, block in enumerate(doc["blocks"]):
                text = "\n".join(block["lines"]).strip()
                if text:
                    source_by_idx[doc["offset"] + i] = text
        if not source_by_idx:
            return {}
        hits = translation_memory.exact_matches(
            list(source_by_idx.values()), lang_pair=f"{lang.code}-ko",
            statuses=_TM_REUSE_STATUSES)
        filled = {idx: hits[text].split("\n")
                  for idx, text in source_by_idx.items() if text in hits}
        if filled:
            emit({"event": "tmReuse", "blocks": len(filled)})
        return filled
    except Exception as e:
        emit({"event": "tmReuseFailed", "error": str(e)})
        return {}


def _record_translation_memory(spec: dict, lang: SourceLanguage, succeeded, opts: Options,
                               emit: Callable[[dict], None]) -> None:
    """성공한 청크의 블록 쌍을 코퍼스 단위 번역 메모리에 적재한다 (인수인계 §5-1).

    청크 캐시는 이 파이프라인 안에서만 재사용되는 해시 키 저장소라 문서 간
    연속성에 쓸 수 없다. TM은 정렬된 (원문, 번역) 쌍을 남겨 다음 단계(완전
    일치 재사용, 예시 주입)의 토대가 된다. 캐시에서 나온 청크도 다시 적재한다
    — INSERT OR IGNORE라 중복 비용이 없고, 캐시보다 늦게 생긴 TM에 과거
    실행분이 재실행만으로 채워진다. TM 실패가 번역 실행을 깨서는 안 되므로
    어떤 예외도 여기서 멈춘다.
    """
    try:
        from llm import call_registry
        from runtime_tools import translation_memory

        pairs: list[tuple[str, str]] = []
        block_ids: list[int] = []
        for chunk, got in succeeded:
            for idx, block in chunk:
                lines = got.get(idx)
                if not lines:
                    continue
                pairs.append((
                    "\n".join(block["lines"]),
                    "\n".join(apply_post_edits(lines, spec)),
                ))
                block_ids.append(idx)
        if not pairs:
            return
        profile = call_registry.resolve(lang.feature)
        inserted = translation_memory.record_segments(
            pairs, lang_pair=f"{lang.code}-ko", doc_id=spec.get("id", "unnamed"),
            block_ids=block_ids, provider=profile.provider, model=profile.model)
        emit({"event": "tm", "segments": len(pairs), "inserted": inserted})
    except Exception as e:
        emit({"event": "tmFailed", "error": str(e)})



def reassemble(spec: dict, opts: Options | None = None) -> dict:
    """캐시 레코드를 블록 번호로 직접 조립해 fragment만 다시 쓴다 (LLM 호출 없음).

    run()의 캐시 조회 키에는 provider·model·system_hash가 들어가므로, 모델을
    교체하거나 프롬프트를 고친 뒤에는 옛 문서의 postEdits 손질이 전체 재번역이
    되어 버린다. 발행본 손질은 번역을 새로 만드는 일이 아니라 이미 확정된
    번역을 다시 배치하는 일이므로, 키를 무시하고 레코드의 블록 번호로 조립한다.
    캐시에는 청킹·프롬프트를 바꿔 가며 만든 이력 레코드가 섞여 있을 수 있다.
    블록 집합이 현재 청킹의 어느 청크와도 일치하지 않는 레코드는 다른 청킹
    시절의 고아이므로 배제한다 — 포스펠로프 재조립에서 last-wins가 고아
    레코드의 대체 번역을 집어 발행문이 바뀐 사례. 같은 청크의 레코드가 여럿
    이면 나중 레코드(재시도 최신본)가 이긴다. 비어 있는 블록이 남으면 고아
    레코드로 채우되 경고하고, 그래도 남으면 구멍 난 문서 대신 예외를 낸다.
    """
    opts = opts or Options()
    if spec.get("frozen"):
        raise SpecError(
            f"{spec.get('id')}: 완료된 스펙이라 재실행이 막혀 있다 (frozen: {spec['frozen']})")
    prepared = plan(spec, opts)
    docs, chunks, lang = prepared["_docs"], prepared["_chunks"], prepared["_lang"]

    chunk_sets = {frozenset(idx for idx, b in c if b["lines"]) for c in chunks}
    records = list(Cache(_cache_path(spec, opts.cache_path)).data.values())
    translated: dict[int, list[str]] = {}
    for rec in records:
        blocks = {int(k): v for k, v in rec["blocks"].items()}
        if frozenset(blocks) in chunk_sets:
            translated.update(blocks)

    expected = {idx for c in chunks for idx, b in c if b["lines"]}
    orphan_filled = sorted(expected - translated.keys())
    if orphan_filled:
        for rec in records:
            for k, v in rec["blocks"].items():
                if int(k) in orphan_filled:
                    translated[int(k)] = v
        print(f"경고: 블록 {orphan_filled}는 현재 청킹과 일치하는 레코드가 없어 "
              f"고아 레코드에서 채웠다. --max-chars가 번역 당시와 같은지 확인할 것.",
              file=sys.stderr)
    # run()과 같은 우선순위: 검수 등급 TM 세그먼트가 모델 번역을 덮는다.
    # 이걸 빠뜨리면 발행문이 TM 문구에서 캐시 문구로 슬그머니 바뀐다.
    translated.update(_tm_prefill(docs, lang, lambda _e: None))

    missing = sorted(expected - translated.keys())
    if missing:
        raise SpecError(
            f"{spec.get('id')}: 캐시에 없는 블록 {missing} — 재조립으로는 채울 수 없다. "
            "새로 번역해야 하는 변경이면 run()을 쓸 것.")

    out_path = opts.out_path or Path(spec["output"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    html = assemble(spec, docs, translated)
    out_path.write_text(html, encoding="utf-8")
    return {"output": str(out_path), "bytes": out_path.stat().st_size,
            "blocks": len(translated),
            "strayCyrillic": stray_cyrillic(html, spec.get("allowedCyrillic"), lang)}


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
    lang = prepared["_lang"]

    cache = Cache(_cache_path(spec, opts.cache_path))

    tm_filled = _tm_prefill(docs, lang, emit)
    # TM이 청크의 모든(비어 있지 않은) 블록을 덮으면 그 청크는 모델도 캐시도
    # 필요 없다. 일부만 덮인 청크는 통째로 돌린다 — 덮인 블록만 빼고 청크를
    # 다시 자르면 프롬프트가 달라져 기존 청크 캐시가 전부 무효가 되고,
    # 재사용이 비용을 줄이는 게 아니라 늘리게 된다.
    runnable = [c for c in chunks
                if not all((idx in tm_filled) or not b["lines"] for idx, b in c)]
    prepared_chunks = [_prepare_chunk(c, glossary, opts, lang) for c in runnable]
    pending = sum(1 for c, (_, key) in zip(runnable, prepared_chunks)
                  if _cached_blocks(cache, key, c, lang)[0] is None)
    # Re-assembling a fully cached run (a postEdits tweak, a headnote change)
    # makes no API call, so demanding a credential for it would be wrong.
    if pending:
        preflight(opts, lang)
    emit({"event": "plan", "pending": pending,
          "tmReusedBlocks": len(tm_filled),
          "chunksSkipped": len(chunks) - len(runnable),
          **{k: v for k, v in prepared.items() if not k.startswith("_")}})

    stats = Stats()
    started = time.time()

    with ThreadPoolExecutor(max_workers=opts.concurrency) as pool:
        futures = [pool.submit(_translate_chunk, c, glossary, cache, opts, stats,
                               emit, lang, prep)
                   for c, prep in zip(runnable, prepared_chunks)]
        succeeded: list[tuple[list, dict[int, list[str]]]] = []
        failures = []
        for i, (fut, chunk) in enumerate(zip(futures, runnable), 1):
            span = [chunk[0][0], chunk[-1][0]]
            try:
                succeeded.append((chunk, fut.result()))
            except Exception as e:
                # One bad chunk must not discard the 40 good ones: they are
                # already cached, so a re-run costs only the failures.
                failures.append({"blocks": span, "error": str(e)})
                emit({"event": "chunkFailed", "blocks": span, "error": str(e)})
            emit({"event": "chunk", "done": i, "total": len(futures)})

    translated: dict[int, list[str]] = {}
    for _, r in succeeded:
        translated.update(r)
    # 부분 덮임 청크에서는 모델 출력과 TM이 겹친다. 검수 등급 TM이 이긴다.
    translated.update(tm_filled)

    # 실패한 청크가 있어도 성공분은 유효한 정렬 쌍이므로 먼저 적재한다.
    # (TM에서 온 블록은 모델 출력이 아니므로 다시 적재하지 않는다.)
    _record_translation_memory(spec, lang, succeeded, opts, emit)

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
    result["strayCyrillic"] = stray_cyrillic(html, spec.get("allowedCyrillic"), lang)
    emit({"event": "done", **result})
    return result
