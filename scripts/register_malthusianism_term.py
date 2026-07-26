#!/usr/bin/env python3
"""Register the 맬서스주의 glossary entry.

The 임금철칙 entry already rests on Malthus — Marx's second line of attack is
that if you take Lassalle's law with its Malthusian substantiation, the law
survives the abolition of wage labour a hundred times over — and the glossary
had no entry for the substantiation. This closes that.

Written against the primary texts, not against the received summary. Two points
where the tertiary Korean literature is loose and this entry is not:

- The 'wages fund' doctrine is routinely dated to the late nineteenth century
  and attributed to James Mill and McCulloch. Mill died in 1836 and McCulloch in
  1864; the doctrine is early-to-mid century and was abandoned by J. S. Mill in
  1869. The entry stays with what can be dated.
- Malthus is usually summarised from the 1798 first edition alone. He published
  six editions in his lifetime and the 1803 revision added moral restraint as a
  third check, which is what later divides him from the neo-Malthusians who
  advocated contraception.

No person card is created: the people dictionary is 혁명과 소련의 사람들, and
Malthus sits outside it. He is named in prose; Marx and Engels carry the links.

Term↔term relations go in the companion frontend migration
scripts/migrations/106_commulingo_malthusianism_relations.sql.

Usage:
  bash scripts/run_commulingo_register.sh scripts/register_malthusianism_term.py
  bash scripts/run_commulingo_register.sh scripts/register_malthusianism_term.py --apply
  bash scripts/run_commulingo_register.sh scripts/register_malthusianism_term.py --update --apply
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TERM_ID = "malthusianism"

SOURCES = [
    "https://www.econlib.org/library/Malthus/malPop.html?chapter_num=2 — An Essay on the "
    "Principle of Population (1798), ch. I: 'I think I may fairly make two postulata. First, "
    "That food is necessary to the existence of man. Secondly, That the passion between the "
    "sexes is necessary and will remain nearly in its present state.' And: 'Population, when "
    "unchecked, increases in a geometrical ratio. Subsistence increases only in an arithmetical "
    "ratio.' The same chapter names Godwin and Condorcet as the advocates of perfectibility he "
    "is answering",
    "https://en.wikipedia.org/wiki/An_Essay_on_the_Principle_of_Population — published "
    "anonymously in 1798 against Godwin and Condorcet; the much-enlarged 1803 second edition "
    "adds moral restraint as a third check beside the preventive and positive ones; six "
    "editions in Malthus's lifetime (1798, 1803, 1806, 1807, 1817, 1826); its part in the Poor "
    "Law Amendment Act of 1834; and Darwin's and Wallace's acknowledgement of the sixth edition",
    "https://ko.wikipedia.org/wiki/1834년_신빈민법 — 1834년 개정 빈민법은 '토머스 로버트 "
    "맬서스와 제러미 벤담, \"임금의 철칙\" 등을 이론적 근거로 하였다'. 열등처우의 원칙으로 구제 "
    "수준을 최저 노동 수입에 미치지 못하게 하고 구빈원 외의 구제를 금지했다",
    "https://www.marxists.org/archive/marx/works/1867-c1/ch25.htm — Capital vol. I, ch. 25: "
    "'This is a law of population peculiar to the capitalist mode of production; and in fact "
    "every special historic mode of production has its own special laws of population, "
    "historically valid within its limits'; the surplus population as 'a disposable industrial "
    "reserve army, that belongs to capital quite as absolutely as if the latter had bred it at "
    "its own cost'; 'the general movements of wages are exclusively regulated by the expansion "
    "and contraction of the industrial reserve army'; and footnote 6, which calls the first "
    "edition of the Essay 'a schoolboyish, superficial plagiary of De Foe, Sir James Steuart, "
    "Townsend, Franklin, Wallace, &c.' whose sensation 'was due solely to party interest'",
    "https://www.marxists.org/archive/marx/works/1844/df-jahrbucher/outlines.htm — Engels, "
    "Outlines of a Critique of Political Economy (1844): the Malthusian population theory is "
    "'the crudest, most barbarous theory that ever existed, a system of despair', and 'surplus "
    "population or labour-power is invariably tied up with surplus wealth, surplus capital and "
    "surplus landed property'",
    "https://www.marxists.org/archive/marx/works/1875/gotha/ch02.htm — Marx on the substantiation "
    "of Lassalle's law: 'As Lange already showed, shortly after Lassalle's death, it is the "
    "Malthusian theory of population… But if this theory is correct, then again I cannot abolish "
    "the law even if I abolish wage labor a hundred times over'",
    "https://en.wikipedia.org/wiki/Neo-Malthusianism — neo-Malthusians 'differ from Malthus's "
    "theories mainly in their support for the use of birth control', which Malthus the clergyman "
    "rejected in favour of self-control; Paul Ehrlich's The Population Bomb (1968) and the Club "
    "of Rome's The Limits to Growth (1972) as the twentieth-century revival, and the standard "
    "criticisms from agricultural advance and demographic transition",
]

DEFINITION_KO = (
    "1798년 토머스 로버트 맬서스가 익명으로 낸 『인구론』에서 나온 학설. 인구는 억제되지 않으면 "
    "기하급수로 늘고 생계 수단은 산술급수로 늘 뿐이므로 빈곤과 기아는 제도의 산물이 아니라 자연의 "
    "법칙이라는 것이다. 고드윈과 콩도르세의 인간 완전가능성론을 겨냥해 쓰였고, 1834년 신빈민법의 "
    "이론적 근거가 되었으며, 라살레의 임금철칙도 여기에 기댔다. 마르크스는 초역사적 인구법칙이란 "
    "없고 생산양식마다 자기 인구법칙이 있다고 반박했으며, 엥겔스는 이것을 '지금까지 존재한 가장 "
    "조야하고 야만적인 이론, 절망의 체계'라 불렀다."
)

DEFINITION_EN = (
    "The doctrine from An Essay on the Principle of Population, published anonymously by "
    "Thomas Robert Malthus in 1798. Population unchecked grows in a geometrical ratio while "
    "subsistence grows only in an arithmetical one, so poverty and hunger are laws of nature "
    "rather than products of institutions. It was written against the perfectibility argued by "
    "Godwin and Condorcet, it supplied the theoretical ground for the Poor Law Amendment Act of "
    "1834, and Lassalle's iron law of wages leaned on it. Marx replied that there is no "
    "trans-historical law of population and that every mode of production has its own; Engels "
    "called it 'the crudest, most barbarous theory that ever existed, a system of despair'."
)

BODY_KO = """## 두 개의 공준, 두 개의 비율

논증은 짧다. 맬서스는 두 가지를 공준으로 놓는다. "첫째, 식량은 인간의 생존에 필요하다. 둘째, 양성 간의 정념은 필요하며 거의 지금 상태로 남을 것이다." 여기서 곧바로 결론이 나온다.

"인구는 억제되지 않으면 기하급수로 증가한다. 생계 수단은 산술급수로 증가할 뿐이다."

두 힘의 크기가 다르므로 균형은 강제로 맞춰질 수밖에 없다. 그 강제가 억제(check)다. 초판은 이를 예방적 억제(만혼과 독신)와 적극적 억제(기아·질병·전쟁)로 나눈다. 1803년의 대폭 개정판이 여기에 도덕적 억제를 셋째 항목으로 더한다. 맬서스는 생전에 여섯 판(1798, 1803, 1806, 1807, 1817, 1826)을 냈고, 흔히 요약되는 초판만으로 그를 읽으면 뒤에 나올 신맬서스주의와의 차이를 놓치게 된다.

## 무엇에 맞선 책이었나

초판은 익명으로 나왔고, 겨냥한 상대는 분명했다. 맬서스 자신이 1장에서 고드윈과 콩도르세를 지목한다. 프랑스 혁명이 열어 놓은 인간 완전가능성의 전망 — 제도를 바꾸면 빈곤을 없앨 수 있다는 주장 — 에 대해, 그것은 제도의 문제가 아니라 산술의 문제라고 답한 것이 이 책이다.

이 응답의 구조가 중요하다. 빈곤의 원인을 사회에서 자연으로 옮기면 사회를 바꾸자는 요구는 자동으로 무의미해진다. 이 책이 오래 살아남은 이유도, 마르크스주의가 이 책을 그토록 집요하게 공격한 이유도 여기에 있다.

## 이론이 법이 되었을 때

1834년 개정 빈민법이 그 결과다. 왕립빈민법조사위원회 보고에 기반한 이 법은 맬서스와 벤담, 그리고 임금철칙을 이론적 근거로 삼았다. 핵심 장치는 두 가지였다. 구제 수준을 가장 낮은 노동 수입에도 미치지 못하게 묶는 열등처우의 원칙, 그리고 구빈원 밖에서의 구제 금지. 구호가 인구를 부양해 다시 빈곤을 부른다는 논리를 그대로 제도로 옮긴 것이다.

## 마르크스의 반박 — 인구법칙은 역사적이다

『자본론』 1권 25장은 이 학설을 정면으로 겨눈다. 마르크스가 부정한 것은 과잉인구의 존재가 아니라 그것의 자연성이다.

"이것은 자본주의적 생산양식에 고유한 인구법칙이며, 사실 모든 특수한 역사적 생산양식은 자기 자신의 특수한 인구법칙을 가진다."

과잉인구는 자연이 낳는 것이 아니라 축적이 낳는다. 자본이 기술 구성을 높이며 축적할수록 상대적 과잉인구가 생기고, 그것은 다시 축적의 지렛대가 된다. 마르크스는 그것을 산업예비군이라 부른다. "자본이 자기 비용으로 길러 낸 것이나 다름없이 자본에 속하는 처분 가능한 산업예비군." 그리고 여기서 임금 문제로 되돌아온다. "임금의 일반적 운동은 오로지 산업예비군의 팽창과 수축에 의해 규제된다."

즉 임금을 생존선으로 누르는 힘은 실재하지만 그 힘은 인구의 자연적 압력이 아니라 자본축적의 운동이다. 자연법칙이 아니라 사회관계이므로 바꿀 수 있는 것이 된다.

같은 장의 각주에서 마르크스는 학설사 쪽에서도 칼을 댄다. 초판의 『인구론』은 "디포, 제임스 스튜어트 경, 타운센드, 프랭클린, 월리스 등에 대한 유치하고 피상적인 표절"이며 스스로 생각해 낸 문장이 하나도 없는데, 그럼에도 큰 반향을 일으킨 것은 "오로지 당파적 이해관계 때문"이었다는 것이다.

## 엥겔스의 1844년 판정

엥겔스는 스물넷에 쓴 「국민경제학 비판 개요」에서 이미 결론을 내려 두었다. 맬서스의 인구론은 "지금까지 존재한 가장 조야하고 야만적인 이론, 절망의 체계"라는 것이다. 그가 제시한 반증은 간단하다. 과잉인구는 언제나 과잉의 부, 과잉의 자본, 과잉의 토지 소유와 짝을 이루어 나타난다. 인구가 너무 많은 곳은 생산력 전체가 너무 큰 곳이다.

## 임금철칙으로 이어지는 선

라살레의 임금철칙이 이 학설 위에 서 있다. 임금이 생존 수준으로 되돌아가는 이유를 인구 압력에서 찾았기 때문이다. 『고타강령 비판』 2부에서 마르크스가 놓은 덫이 바로 그 지점이다. 라살레의 법칙을 그의 뜻대로 받아들이면 근거인 맬서스 인구론도 함께 받아들여야 하는데, 그 이론이 옳다면 임금노동을 백 번 폐지해도 법칙은 살아남는다. 법칙이 지배하는 것이 임금노동 체제만이 아니라 모든 사회 체제가 되기 때문이다. 경제학자들이 50년 넘게 사회주의는 빈곤을 없애지 못하고 사회 전면에 고루 퍼뜨릴 뿐이라고 논증해 온 것이 바로 이 논리다.

## 오늘의 맬서스주의

신맬서스주의는 맬서스 본인과 한 가지에서 갈라진다. 성직자였던 맬서스는 피임을 거부하고 자제를 요구했지만, 19세기의 신맬서스주의자들은 산아제한을 주장했다. 결론은 물려받고 수단만 바꾼 것이다.

20세기의 부활은 1968년 폴 에얼릭의 『인구폭탄』과 1972년 로마클럽의 『성장의 한계』로 왔다. 녹색혁명의 농업 생산성 증가와 선진국의 인구 전환이 그 예측을 빗나가게 했다는 것이 표준적인 비판이다.

이 항목이 지금도 필요한 이유는 예측의 성패보다 논증의 형식에 있다. 부족을 자연의 사실로 놓고 분배와 소유를 묻지 않는 논법은 인구를 두고서만 쓰이지 않는다. 식량이든 에너지든 탄소든, 한계를 자연에서 찾을 때마다 같은 물음이 되돌아온다. 누가 무엇을 얼마나 가지고 있느냐를 먼저 묻지 않았을 때 그 한계는 언제나 가장 가난한 쪽의 몫으로 계산된다."""

BODY_EN = """## Two Postulata, Two Ratios

The argument is short. Malthus lays down two postulates: 'First, That food is necessary to the existence of man. Secondly, That the passion between the sexes is necessary and will remain nearly in its present state.' The conclusion follows at once.

'Population, when unchecked, increases in a geometrical ratio. Subsistence increases only in an arithmetical ratio.'

Because the two powers are unequal, the balance has to be forced, and the forcing is done by checks. The first edition divides them into preventive checks (late marriage, celibacy) and positive ones (hunger, disease, war). The much-enlarged second edition of 1803 adds moral restraint as a third. Malthus published six editions in his lifetime — 1798, 1803, 1806, 1807, 1817, 1826 — and reading him from the commonly summarised first edition alone loses what later separates him from the neo-Malthusians.

## What the Book Was Written Against

The first edition appeared anonymously, and its target is stated in chapter I: Godwin and Condorcet. Against the prospect of human perfectibility opened by the French Revolution — the claim that changing institutions can abolish poverty — the book answers that this is not a question of institutions but of arithmetic.

The structure of that answer is what matters. Move the cause of poverty from society to nature and the demand to change society becomes meaningless on its own terms. That is why the book had such a long life, and why Marxism went after it so persistently.

## When the Theory Became Law

The Poor Law Amendment Act of 1834 is the result. Drafted on the report of the Royal Commission, it took Malthus, Bentham and the iron law of wages as its theoretical grounds. Two devices carried it: the principle of less eligibility, holding relief below the lowest labouring income, and the ban on relief outside the workhouse. The reasoning that relief feeds population and so breeds poverty was transcribed directly into institutions.

## Marx's Reply — Laws of Population Are Historical

Chapter 25 of Capital vol. I goes straight at the doctrine. What Marx denies is not that surplus population exists but that it is natural.

'This is a law of population peculiar to the capitalist mode of production; and in fact every special historic mode of production has its own special laws of population, historically valid within its limits.'

Surplus population is produced by accumulation, not by nature. As capital accumulates and raises its technical composition, a relative surplus population forms, and that surplus becomes in turn the lever of accumulation — 'a disposable industrial reserve army, that belongs to capital quite as absolutely as if the latter had bred it at its own cost'. From there the argument returns to wages: 'the general movements of wages are exclusively regulated by the expansion and contraction of the industrial reserve army.'

The force pressing wages toward subsistence is real, then, but it is the movement of accumulation and not the natural pressure of numbers. Being a social relation rather than a law of nature, it can be changed.

In a footnote to the same chapter Marx also settles accounts on the history of doctrine: the Essay in its first form is 'a schoolboyish, superficial plagiary of De Foe, Sir James Steuart, Townsend, Franklin, Wallace, &c.' containing not one sentence thought out by its author, and the sensation it caused 'was due solely to party interest'.

## Engels's Verdict of 1844

Engels had reached his conclusion at twenty-four, in the Outlines of a Critique of Political Economy: the Malthusian population theory is 'the crudest, most barbarous theory that ever existed, a system of despair'. His counter-evidence is simple. Surplus population is invariably tied up with surplus wealth, surplus capital and surplus landed property. A country is overpopulated only where its productive power as a whole is too large.

## The Line Into the Iron Law of Wages

Lassalle's iron law of wages stands on this doctrine, since it locates the reason wages fall back to subsistence in the pressure of numbers. That is exactly where Marx set his trap in Part II of the Critique of the Gotha Programme. Take Lassalle's law in his sense and you must take its Malthusian substantiation with it — and if that theory is correct, the law survives the abolition of wage labour a hundred times over, because it then governs not the wage system alone but every social system. This is the reasoning on which economists had been proving for fifty years that socialism cannot abolish poverty but only spread it evenly across society.

## Malthusianism Now

Neo-Malthusianism parts from Malthus on one point. The clergyman rejected contraception and asked for self-control; the neo-Malthusians of the nineteenth century advocated birth control. The conclusion was inherited and only the means changed.

The twentieth-century revival came with Paul Ehrlich's The Population Bomb (1968) and the Club of Rome's The Limits to Growth (1972). The standard criticism is that the productivity gains of the green revolution and the demographic transition in developed countries falsified the predictions.

The entry earns its place less through the record of those predictions than through the form of the argument. Treating scarcity as a fact of nature and declining to ask about distribution and ownership is a move that is not reserved for population. Food, energy, carbon: every time a limit is located in nature, the same question comes back. Where nobody asks first who holds how much, the limit is always charged to the poorest."""

PATCH = {
    "term": {"ko": "맬서스주의", "en": "Malthusianism"},
    "original": "Malthusianism",
    "period": {"ko": "1798년~", "en": "1798–present"},
    "startYear": 1798,
    "category": "theory",
    "definition": {"ko": DEFINITION_KO, "en": DEFINITION_EN},
    "body": {"ko": BODY_KO, "en": BODY_EN},
    "aliases": {
        # '맬서스 인구론' is registered alongside '인구론' so the longer string
        # wins the longest-first alternation and the whole phrase links, rather
        # than the link opening in the middle of it.
        "ko": ["맬서스주의", "맬더스주의", "맬서스 인구론", "인구론", "신맬서스주의"],
        "en": ["Malthusianism", "Malthusian", "neo-Malthusianism", "principle of population"],
    },
    # Malthus has no card: the people dictionary is 혁명과 소련의 사람들 and he
    # sits outside it. The two who carry the argument here do have cards.
    "people": ["karl-marx", "friedrich-engels"],
    "sources": SOURCES,
}


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    parser.add_argument(
        "--update", action="store_true",
        help="rewrite the entry if it already exists, sending the whole patch",
    )
    args = parser.parse_args()

    print(f"term: {TERM_ID}")
    print(f"  headword      {PATCH['term']['ko']} / {PATCH['term']['en']}")
    print(f"  category      {PATCH['category']}   period {PATCH['period']['ko']}")
    print(f"  definition ko {len(DEFINITION_KO)}/400")
    print(f"  definition en {len(DEFINITION_EN)}/900")
    print(f"  body ko/en    {len(BODY_KO)} / {len(BODY_EN)} chars")
    print(f"  aliases       ko {len(PATCH['aliases']['ko'])}, en {len(PATCH['aliases']['en'])}")
    print(f"  people        {', '.join(PATCH['people'])}")
    print(f"  sources       {len(SOURCES)}")

    if len(DEFINITION_KO) > 400 or len(DEFINITION_EN) > 900:
        print("ABORT: definition exceeds the card limit", file=sys.stderr)
        return 1

    if not args.apply:
        print("\ndry run; pass --apply to write")
        return 0

    from runtime_tools.commulingo_people import _exec_commulingo_write

    result = await _exec_commulingo_write(
        "term", "update" if args.update else "create", TERM_ID, SOURCES, PATCH, 0.95,
    )
    print("\n" + result)
    return 0 if not (result.startswith("Error:") or '"error"' in result) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
