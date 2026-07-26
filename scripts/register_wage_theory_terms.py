#!/usr/bin/env python3
"""Register the 임금기금설 and 한계생산력설 glossary entries.

The two bourgeois wage theories that stand on either side of 임금철칙 in time:
the classical one it was built on top of, and the neoclassical one that replaced
both. With 맬서스주의 already registered, this completes the run.

  wages-fund-doctrine            임금기금설      1820년대–1869
  marginal-productivity-theory   한계생산력설    1890년대~

Both flat, neither nested: 임금철칙 hangs under 고타강령 비판 because Marx
dissects it there, while these two are not from that text.

Written against the primary texts. Three corrections to the received Korean
summaries, including the 노동자의 책 dictionary entry that prompted these:

- The wages fund is not a late-nineteenth-century doctrine of James Mill and
  McCulloch. Mill died in 1836, McCulloch in 1864; the canonical statement is
  J. S. Mill's Principles of 1848, and the doctrine was dead by 1869 — killed by
  its own author.
- Marx's demolition of it is not in Capital but in the June 1865 address to the
  General Council of the International against Citizen Weston, published as
  Value, Price and Profit. Weston's two premises are named there and answered
  one at a time.
- The Cambridge capital controversy is usually reported as having refuted
  marginal productivity theory outright. What reswitching and capital reversing
  broke is the AGGREGATE version — the production function in one homogeneous
  capital — and Samuelson conceded exactly that much and no more. The entry says
  what was conceded rather than overclaiming, because overclaiming here is how
  the argument gets waved away.

Term↔term relations go in the companion frontend migration
scripts/migrations/107_commulingo_wage_theory_relations.sql.

Usage:
  bash scripts/run_commulingo_register.sh scripts/register_wage_theory_terms.py
  bash scripts/run_commulingo_register.sh scripts/register_wage_theory_terms.py --apply
  bash scripts/run_commulingo_register.sh scripts/register_wage_theory_terms.py --update --apply
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MIA = "https://www.marxists.org"

SRC_WAGE_FUND = (
    "https://en.wikipedia.org/wiki/Wage_fund_doctrine — wages equal capital divided by "
    "population, in McCulloch's formulation 'Laborers are everywhere the divisor, capital the "
    "dividend'; J. S. Mill's Principles of Political Economy (1848) as the definitive treatment; "
    "and Mill's 1869 recantation after reading W. T. Thornton's On Labour, calling the doctrine "
    "a 'prevailing and mischievous error' once he saw that the fund could be supplemented from "
    "income the employer would otherwise save or spend"
)
SRC_THORNTON = (
    "https://en.wikipedia.org/wiki/William_Thomas_Thornton — On Labour (1869) was reviewed twice "
    "in The Fortnightly Review by John Stuart Mill"
)
SRC_VPP_1 = (
    f"{MIA}/archive/marx/works/1865/value-price-profit/ch01.htm — Marx to the General Council of "
    "the International, June 1865: 'Citizen Weston's argument rested, in fact, upon two premises: "
    "firstly, the amount of national production is a fixed thing, a constant quantity or "
    "magnitude… secondly, that the amount of real wages… is a fixed amount, a constant "
    "magnitude.' And the reply: 'The amount or magnitude of national production changes "
    "continuously. It is not a constant but a variable magnitude'"
)
SRC_VPP_3 = (
    f"{MIA}/archive/marx/works/1865/value-price-profit/ch03.htm — the conclusion: workers 'are "
    "fighting with effects, but not with the causes of those effects; that they are retarding the "
    "downward movement, but not changing its direction; that they are applying palliatives, not "
    "curing the malady', and 'Instead of the conservative motto: \"A fair day's wage for a fair "
    "day's work!\" they ought to inscribe on their banner the revolutionary watchword: "
    "\"Abolition of the wages system!\"'"
)
SRC_CAPITAL_25 = (
    f"{MIA}/archive/marx/works/1867-c1/ch25.htm — Capital vol. I, ch. 25: 'the general movements "
    "of wages are exclusively regulated by the expansion and contraction of the industrial "
    "reserve army', which puts the determinant in accumulation rather than in any fixed fund"
)
SRC_KO_MARGINAL = (
    "https://ko.wikipedia.org/wiki/한계생산력 — 한국어 위키백과: 분배이론으로서의 '한계생산력설'은 "
    "1890년대에 윅스티드·바로네·발라·클라크가 정식화했고, 클라크만이 이를 국민생산물의 사회적 "
    "분배원리로 거시적으로 사용했으며 나머지는 기업의 생산·분배 이론으로 다루었다. 빅셀을 거쳐 "
    "1930년대 힉스 등이 오늘의 형태로 완성했다"
)
SRC_CLARK = (
    "https://en.wikipedia.org/wiki/John_Bates_Clark — The Distribution of Wealth (1899) and its "
    "political message: '[W]hat a social class gets is, under natural law, what it contributes to "
    "the general output of industry.' Clark's treatment of capital as homogeneous 'jelly' rather "
    "than distinct produced goods is what later opened the Cambridge capital controversy"
)
SRC_CCC = (
    "https://en.wikipedia.org/wiki/Cambridge_capital_controversy — Joan Robinson, 'The Production "
    "Function and the Theory of Capital', Review of Economic Studies 21:2 (1953), on valuing "
    "capital requiring the rate of profit that the production function is meant to determine; "
    "reswitching and capital reversing showing 'no simple (monotonic) relationship between the "
    "nature of the techniques of production used and the rate of profit'; and Samuelson's "
    "'Summing Up', Quarterly Journal of Economics 80 (1966), p. 568: 'the simple tale told by "
    "Jevons, Böhm-Bawerk, Wicksell and other neoclassical writers… cannot be universally valid'"
)

# ── 임금기금설 ────────────────────────────────────────────────────────

WAGE_FUND = {
    "id": "wages-fund-doctrine",
    "sources": [SRC_WAGE_FUND, SRC_THORNTON, SRC_VPP_1, SRC_VPP_3, SRC_CAPITAL_25],
    "patch": {
        "term": {"ko": "임금기금설", "en": "Wages-Fund Doctrine"},
        "original": "wages-fund doctrine",
        "period": {"ko": "1820년대–1869", "en": "1820s–1869"},
        "startYear": 1820,
        "endYear": 1869,
        "category": "theory",
        "aliases": {
            "ko": ["임금기금설", "임금기금론", "임금기금"],
            "en": ["wages-fund doctrine", "wage fund doctrine", "wages fund theory"],
        },
        "people": ["karl-marx"],
        "definition": {
            "ko": (
                "어느 시점에나 임금으로 지불될 수 있는 자본의 총액이 정해져 있고 평균임금은 그 기금을 "
                "노동자 수로 나눈 값이라는 고전파 정치경제학의 교리. 맥컬록의 표현대로 '노동자는 "
                "어디서나 제수이고 자본은 피제수'다. 결론은 노동조합 무용론이었다. 기금이 고정되어 "
                "있으니 한 집단의 임금 인상은 다른 집단의 삭감이나 실업으로만 상쇄된다는 것이다. "
                "1848년 J.S. 밀의 『정치경제학 원리』가 정본을 세웠고, 같은 밀이 1869년 이를 '널리 "
                "퍼진 해로운 오류'라 부르며 스스로 거두어들였다."
            ),
            "en": (
                "The classical doctrine that at any moment a fixed sum of capital is available to "
                "pay wages, so that the average wage is that fund divided by the number of "
                "workers — in McCulloch's formulation, 'Laborers are everywhere the divisor, "
                "capital the dividend'. The conclusion was that trade unions are useless: with "
                "the fund fixed, a rise won by one group can only be offset by cuts or "
                "unemployment among others. J. S. Mill's Principles of Political Economy (1848) "
                "gave it its definitive statement, and the same Mill withdrew it in 1869 as a "
                "'prevailing and mischievous error'."
            ),
        },
        "body": {
            "ko": """## 하나의 나눗셈

교리 전체가 분수 하나로 요약된다. 분모는 노동자 수, 분자는 임금 지불에 쓰이는 자본. 맥컬록이 그것을 문장으로 만들었다. "노동자는 어디서나 제수이고 자본은 피제수다."

전제는 분자가 단기에 고정되어 있다는 것이다. 유동자본 가운데 임금으로 갈 몫은 이미 정해져 있어서, 올해 노동자 전체가 받을 총액은 협상 전에 결정되어 있다는 것이다.

## 이 교리가 하려던 일

여기서 나오는 결론은 하나다. 노동조합은 임금 총액을 바꿀 수 없다. 한 부문이 더 받아내면 다른 부문이 덜 받거나 일자리를 잃는다. 파업은 몫의 재배치일 뿐이고, 잘해야 노동자끼리의 재분배이며, 잘못하면 실업이다.

이것은 학설이기 전에 논거였다. 19세기 중반 영국에서 노동조합의 합법화와 임금 투쟁이 정치 쟁점이 되었을 때, 이 분수는 그 투쟁이 산술적으로 무의미하다는 증명서로 쓰였다. 1848년 J.S. 밀의 『정치경제학 원리』가 그 정본을 마련했다.

## 마르크스가 웨스턴에게 답한 자리

1865년 6월, 국제노동자협회 총평의회에서 존 웨스턴이라는 회원이 정확히 이 논법을 폈다. 임금 인상은 무익하다는 것이었다. 마르크스가 두 차례에 걸쳐 답한 것이 뒷날 『임금, 가격, 이윤』으로 나온 강연이다.

마르크스는 먼저 상대의 전제를 두 개로 분해한다. "웨스턴 시민의 논증은 사실 두 가지 전제에 기대고 있다. 첫째, 국민적 생산의 총량은 고정된 것, 수학자들의 말로 하면 상수라는 것. 둘째, 실질임금의 총액, 곧 그것으로 살 수 있는 상품량으로 잰 임금의 총액이 고정된 양이라는 것."

첫째 전제는 사실이 아니다. "국민적 생산의 양은 끊임없이 변한다. 그것은 상수가 아니라 변수다." 자본축적과 노동생산력이 계속 변하기 때문이다. 둘째 전제는 증명 없이 결론을 미리 놓은 것에 지나지 않는다. 임금 총액이 고정되어 있다는 것은 밝혀야 할 명제이지 출발점이 아니다.

그런데 마르크스는 여기서 멈추지 않는다. 임금 투쟁이 가능하다는 것을 보인 다음, 그것의 한계를 곧바로 덧붙인다. 노동자들은 "결과와 싸우고 있지 그 결과의 원인과 싸우고 있는 것이 아니며, 하강 운동을 늦추고 있을 뿐 그 방향을 바꾸고 있지는 않고, 병을 고치는 것이 아니라 완화제를 쓰고 있다." 그렇다고 싸움을 그만두면 "구제 불능의 부서진 무리"로 떨어진다. 결론은 유명한 두 구호의 교체다. "공정한 하루 노동에 공정한 하루 임금"이라는 보수적 표어 대신 "임금제도의 폐지"를 깃발에 새겨야 한다는 것.

『자본론』 1권 25장은 같은 자리를 다른 각도에서 정리한다. 임금의 일반적 운동을 규제하는 것은 고정된 기금이 아니라 산업예비군의 팽창과 수축이다. 규정하는 것은 정해진 액수가 아니라 축적의 운동이다.

## 저자가 스스로 거두어들이다

교리를 죽인 것은 반대파가 아니라 그 정본을 쓴 사람이었다. 1869년 J.S. 밀은 W.T. 손턴의 『노동론』을 계기로 임금기금설을 "널리 퍼진 해로운 오류"라 부르며 철회했다. 그는 이 책을 『포트나이틀리 리뷰』에 두 차례 서평했다. 기금이라는 것이 애초에 고정되어 있지 않으며, 고용주가 저축하거나 자기 소비에 쓸 소득에서 얼마든지 보충될 수 있다는 점을 인정한 것이다.

## 왜 사라지지 않았나

교과서에서는 사라졌지만 논법은 남았다. 정해진 파이를 가정하고 시작해 그 파이가 정해져 있으니 요구는 무의미하거나 해롭다고 결론짓는 방식 말이다. 최저임금을 올리면 일자리가 그만큼 준다는 주장, 어느 부문의 임금 인상은 다른 부문의 몫을 빼앗는 것이라는 주장이 같은 자리에 선다.

마르크스가 웨스턴에게 요구한 것은 그 반대 주장을 외치는 것이 아니라 전제를 밝히라는 것이었다. 파이는 정말 고정되어 있는가, 고정되어 있다면 무엇이 그것을 고정시키는가. 이 두 질문을 통과하지 못하는 분수는 계산이 아니라 결론이다.""",
            "en": """## One Division

The whole doctrine reduces to a fraction. The denominator is the number of workers, the numerator the capital laid out on wages. McCulloch put it in a sentence: 'Laborers are everywhere the divisor, capital the dividend.'

The premise is that the numerator is fixed in the short run. The part of circulating capital that will go to wages is already determined, so the total the working class will receive this year is settled before any bargaining begins.

## The Work the Doctrine Was Doing

One conclusion follows. Trade unions cannot change the total. If one trade extracts more, another receives less or loses its jobs. A strike redistributes; at best among workers, at worst into unemployment.

This was an argument before it was a theory. When the legalisation of unions and the struggle over wages became political questions in mid-century Britain, the fraction served as a certificate that the struggle was arithmetically pointless. J. S. Mill's Principles of Political Economy (1848) gave it its definitive form.

## Where Marx Answered Weston

In June 1865 a member of the General Council of the International, John Weston, put exactly this argument: raising wages is futile. Marx's reply, given in two sittings, was published later as Value, Price and Profit.

He begins by taking the premises apart: 'Citizen Weston's argument rested, in fact, upon two premises: firstly, the amount of national production is a fixed thing, a constant quantity or magnitude, as the mathematicians would say; secondly, that the amount of real wages, that is to say, of wages as measured by the quantity of the commodities they can buy, is a fixed amount, a constant magnitude.'

The first is simply untrue. 'The amount or magnitude of national production changes continuously. It is not a constant but a variable magnitude' — because accumulation and the productive powers of labour keep changing. The second assumes what it has to prove: that the wage total is fixed is the proposition at issue, not a starting point.

Marx does not stop there. Having shown that wage struggle is possible, he states its limits at once. Workers 'are fighting with effects, but not with the causes of those effects; that they are retarding the downward movement, but not changing its direction; that they are applying palliatives, not curing the malady.' Yet to give up the fight would leave them 'degraded to one level mass of broken wretches past salvation'. The conclusion is the exchange of two slogans: instead of the conservative 'A fair day's wage for a fair day's work!', the banner should read 'Abolition of the wages system!'

Chapter 25 of Capital settles the same ground from another angle. What regulates the general movement of wages is not a fixed fund but the expansion and contraction of the industrial reserve army — not a sum set in advance, but the movement of accumulation.

## Withdrawn by Its Own Author

The doctrine was killed not by its opponents but by the man who had written its definitive statement. In 1869, prompted by W. T. Thornton's On Labour — which he reviewed twice in The Fortnightly Review — J. S. Mill withdrew the wages fund as a 'prevailing and mischievous error', conceding that the fund is not fixed at all and can be supplemented out of income the employer would otherwise have saved or spent.

## Why It Never Went Away

It left the textbooks and kept the reasoning: assume a fixed pie, then conclude from its fixity that the demand is pointless or harmful. Raise the minimum wage and jobs disappear one for one; a rise in one sector's wages comes out of another's share. Same fraction, new clothes.

What Marx demanded of Weston was not the opposite assertion but the premises. Is the pie in fact fixed, and if it is, what fixes it? A fraction that cannot get past those two questions is not a calculation but a conclusion.""",
        },
    },
}

# ── 한계생산력설 ──────────────────────────────────────────────────────

MARGINAL = {
    "id": "marginal-productivity-theory",
    "sources": [SRC_KO_MARGINAL, SRC_CLARK, SRC_CCC, SRC_CAPITAL_25],
    "patch": {
        "term": {"ko": "한계생산력설", "en": "Marginal Productivity Theory"},
        "original": "marginal productivity theory of distribution",
        "period": {"ko": "1890년대~", "en": "1890s–present"},
        "startYear": 1890,
        "category": "theory",
        "aliases": {
            "ko": ["한계생산력설", "한계생산력 이론", "한계생산성 이론"],
            "en": [
                "marginal productivity theory",
                "marginal productivity theory of distribution",
            ],
        },
        "people": ["karl-marx"],
        "definition": {
            "ko": (
                "각 생산요소는 그것을 한 단위 더 넣었을 때 늘어나는 생산물의 가치만큼을 받는다는 분배 "
                "이론. 1890년대에 윅스티드·바로네·발라·클라크가 정식화했고 빅셀을 거쳐 1930년대 힉스 "
                "등이 오늘의 형태로 다듬었다. 나머지가 기업 수준의 요소수요 이론으로 다룬 것을 클라크는 "
                "국민생산물의 사회적 분배 원리로 확장해, 한 사회 계급이 받는 것은 자연법칙에 따라 그 "
                "계급이 산업의 총산출에 기여한 것이라고 했다. 1950~60년대 케임브리지 자본 논쟁은 총량 "
                "자본을 가치로 재려면 이윤율을 먼저 알아야 한다는 순환을 드러냈고, 새뮤얼슨은 1966년 "
                "신고전파의 그 이야기가 보편타당하지는 않다고 인정했다."
            ),
            "en": (
                "The theory that each factor of production receives the value of the output added "
                "by its last unit. Formulated in the 1890s by Wicksteed, Barone, Walras and Clark, "
                "carried on by Wicksell and given its present form by Hicks and others in the "
                "1930s. Where the others treated it as a theory of factor demand at the level of "
                "the firm, Clark extended it into a principle of social distribution: what a "
                "social class gets is, under natural law, what it contributes to the general "
                "output of industry. The Cambridge capital controversy of the 1950s and 60s "
                "exposed the circularity in measuring aggregate capital — its value presupposes "
                "the rate of profit the theory is meant to determine — and in 1966 Samuelson "
                "conceded that the neoclassical tale 'cannot be universally valid'."
            ),
        },
        "body": {
            "ko": """## 명제

이윤을 극대화하는 기업은 어떤 생산요소든 그 요소의 한계생산력 가치가 그 요소의 가격과 같아지는 지점까지 쓴다. 노동에 적용하면 임금은 마지막으로 고용된 노동자가 더한 생산물의 가치와 같아진다. 요소를 더 넣을수록 한계생산력이 체감한다는 가정이 이 결론을 떠받친다.

여기까지는 기업의 요소 수요에 관한 기술적 명제다. 이 형태로는 임금 결정의 한쪽 면, 곧 수요 측면을 서술할 뿐이며 임금이 왜 그 수준인지를 다 설명하지도 않는다. 1890년대에 이것을 정식화한 윅스티드·바로네·발라는 대체로 그 선에서 다루었고, 일반균형 이론 안에 넣으려 했다.

## 클라크의 확장 — 기술에서 정당화로

문제는 확장에서 생긴다. 존 베이츠 클라크는 이것을 기업 이론이 아니라 국민생산물의 사회적 분배 원리로 사용했다. 『부의 분배』(1899)의 정치적 메시지는 한 문장으로 요약된다.

"한 사회 계급이 받는 것은, 자연법칙에 따라, 그 계급이 산업의 총산출에 기여한 것이다."

이 문장에서 두 가지 일이 한꺼번에 일어난다. 첫째, 서술이 규범이 된다. 각자가 받는 것이 각자가 만든 것이라면 현재의 분배는 정의롭다. 둘째, 자본이 노동과 나란히 '기여하는 행위자'가 된다. 자본이 자기 한계생산물을 받아 간다면 무상으로 취득되는 몫, 곧 잉여가치는 정의상 존재하지 않게 된다. 착취를 반박한 것이 아니라 용어에서 지운 것이다.

## 케임브리지 자본 논쟁

클라크의 확장에는 기술적 대가가 따랐다. 사회 전체의 분배를 이렇게 설명하려면 이질적인 기계·건물·재고를 하나의 크기로 합산한 '자본'이 있어야 한다. 클라크는 자본을 균질한 젤리처럼 다루었다.

1953년 조앤 로빈슨이 그 지점을 짚었다. 자본을 가치로 재려면 이윤율을 먼저 알아야 하는데, 이윤율은 바로 그 생산함수가 결정하기로 되어 있는 값이다. 순환이다. 이어 스라파 계열의 작업에서 재전환과 자본 역전이 나왔다. 이윤율이 오르내릴 때 채택되는 기술의 자본집약도가 단조롭게 따라 움직이지 않으며, 높은 이윤율에서 더 자본집약적인 기술이 선택되는 경우가 생긴다는 것이다. 요소 가격이 요소의 희소성과 생산성을 반영한다는 그림이 여기서 깨진다.

1966년 폴 새뮤얼슨은 『계간 경제학』에 「총괄」을 실어 인정했다. "제번스, 뵘바베르크, 빅셀을 비롯한 신고전파 저자들이 들려준 그 단순한 이야기는 보편타당할 수 없다."

무엇이 인정되었는지를 정확히 볼 필요가 있다. 무너진 것은 총량 자본을 쓰는 집계적 판본이다. 기업 수준의 한계 분석 자체가 이 논쟁으로 폐기된 것은 아니며, 논쟁의 사정거리를 두고는 지금도 견해가 갈린다. 다만 클라크가 하려던 일 — 사회 계급들 사이의 분배를 각자의 생산 기여로 설명하고 정당화하는 일 — 은 정확히 무너진 그 판본을 필요로 한다.

## 다른 질문 두 개

마르크스주의 쪽의 반론은 계산이 아니라 질문의 구분에 있다. 무엇이 생산에 기여하는가와 누가 무엇을 취득하는가는 다른 질문이다. 기계가 생산에 필요하다는 것과 기계의 소유자가 그 몫을 가져야 한다는 것 사이에는 소유라는 사회적 제도가 놓여 있다. 한계생산력설은 그 제도를 자연법칙이라 부르며 건너뛴다.

『자본론』이 임금 문제에 대해 내놓은 대답은 다른 쪽에 있다. 임금의 일반적 운동을 규제하는 것은 마지막 노동자의 생산성이 아니라 산업예비군의 팽창과 수축, 곧 축적의 운동이다.

## 지금 이 이론이 서 있는 자리

한계생산력설은 오늘날 노동경제학의 기본 문법이다. 최저임금을 올리면 한계생산성이 그에 못 미치는 일자리가 사라진다는 논증, 자동화가 노동의 한계생산성을 낮춘다는 논증, 임금 격차가 생산성 격차의 반영이라는 논증이 모두 이 문법으로 쓰인다.

그 문법을 쓰는 것과 그것이 분배를 정당화한다고 말하는 것은 다른 일이다. 클라크의 문장에서 '자연법칙에 따라'를 빼면 남는 것은 특정 조건 아래 기업이 요소를 얼마나 쓰는지에 관한 서술이고, 그것을 빼지 않으면 남는 것은 지금의 분배가 옳다는 주장이다. 두 가지를 갈라 읽는 것이 이 항목의 용도다.""",
            "en": """## The Proposition

A profit-maximising firm uses any factor up to the point where the value of that factor's marginal product equals its price. Applied to labour: the wage equals the value of the output added by the last worker hired. The conclusion rests on the assumption that marginal productivity diminishes as more of a factor is added.

So far this is a technical proposition about factor demand. In that form it describes one side of wage determination and does not by itself explain why wages stand where they do. Wicksteed, Barone and Walras, who formulated it in the 1890s, largely kept it there and worked it into general equilibrium theory.

## Clark's Extension — From Technique to Justification

The trouble comes with the extension. John Bates Clark used it not as a theory of the firm but as a principle for the social distribution of the national product. The political message of The Distribution of Wealth (1899) reduces to one sentence:

'What a social class gets is, under natural law, what it contributes to the general output of industry.'

Two things happen in that sentence at once. A description becomes a norm: if each receives what each makes, the existing distribution is just. And capital becomes a contributing agent alongside labour: if capital draws its own marginal product, then a share appropriated without payment — surplus value — does not exist by definition. Exploitation is not refuted; it is spelled out of the vocabulary.

## The Cambridge Capital Controversy

Clark's extension carried a technical price. To explain distribution across society this way you need 'capital' as a single magnitude aggregating heterogeneous machines, buildings and stocks. Clark handled capital as homogeneous jelly.

In 1953 Joan Robinson put her finger on it: valuing capital requires knowing the rate of profit, which is the very quantity the production function is supposed to determine. That is a circle. Work in Sraffa's line then produced reswitching and capital reversing — the capital intensity of the techniques chosen does not move monotonically with the rate of profit, and a higher profit rate can go with a more capital-intensive technique. The picture in which factor prices register scarcity and productivity breaks down there.

In 1966 Paul Samuelson conceded it in 'A Summing Up' in the Quarterly Journal of Economics: 'the simple tale told by Jevons, Böhm-Bawerk, Wicksell and other neoclassical writers… cannot be universally valid.'

It is worth being exact about what was conceded. What broke is the aggregate version that needs a single quantity of capital. Marginal analysis at the level of the firm was not abolished by the controversy, and the reach of the result is still argued over. But what Clark set out to do — explain and justify the distribution between social classes by each one's contribution to production — requires precisely the version that broke.

## Two Different Questions

The Marxist objection is not arithmetical but a separation of questions. What contributes to production and who appropriates what are two different questions. Between the fact that a machine is necessary to production and the claim that the machine's owner should receive its product stands the social institution of ownership. Marginal productivity theory steps over that institution by calling it a law of nature.

Capital answers the wage question from the other side: the general movement of wages is regulated not by the productivity of the last worker but by the expansion and contraction of the industrial reserve army — by the movement of accumulation.

## Where the Theory Stands Now

Marginal productivity is the working grammar of labour economics today. The argument that raising the minimum wage destroys jobs whose marginal product falls below it, the argument that automation lowers the marginal product of labour, the argument that wage gaps register productivity gaps — all are written in it.

Using that grammar and claiming that it justifies the distribution are different acts. Delete 'under natural law' from Clark's sentence and what remains is a description of how much of a factor a firm uses under stated conditions; leave it in and what remains is the claim that the present distribution is right. Reading those two apart is what this entry is for.""",
        },
    },
}

TERMS = [WAGE_FUND, MARGINAL]


def report(entry: dict) -> list[str]:
    problems: list[str] = []
    patch = entry["patch"]
    definition, body = patch["definition"], patch["body"]
    print(f"\nterm: {entry['id']}")
    print(f"  headword      {patch['term']['ko']} / {patch['term']['en']}")
    print(f"  category      {patch['category']}   period {patch['period']['ko']}")
    for lang, limit in (("ko", 400), ("en", 900)):
        n = len(definition.get(lang) or "")
        flag = "  OVER" if n > limit else ""
        print(f"  definition {lang}  {n:>4}/{limit}{flag}")
        if flag:
            problems.append(f"{entry['id']}.definition.{lang} exceeds the card limit")
    print(f"  body ko/en    {len(body['ko']):>5} / {len(body['en'])} chars")
    print(f"  aliases       ko {len(patch['aliases']['ko'])}, en {len(patch['aliases']['en'])}")
    print(f"  people        {', '.join(patch.get('people') or []) or '—'}")
    print(f"  sources       {len(entry['sources'])}")
    return problems


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--only", help="register a single term id")
    parser.add_argument(
        "--update", action="store_true",
        help="rewrite entries that already exist, sending the whole patch",
    )
    args = parser.parse_args()

    entries = [t for t in TERMS if not args.only or t["id"] == args.only]
    problems: list[str] = []
    for entry in entries:
        entry["patch"]["sources"] = entry["sources"]
        problems.extend(report(entry))

    if problems:
        print("\nABORT:")
        for problem in problems:
            print(f"  {problem}")
        return 1

    if not args.apply:
        print("\ndry run; pass --apply to write")
        return 0

    from runtime_tools.commulingo_people import _exec_commulingo_write

    action = "update" if args.update else "create"
    failed = 0
    for entry in entries:
        result = await _exec_commulingo_write(
            "term", action, entry["id"], entry["sources"], entry["patch"], 0.95,
        )
        print(f"\n{entry['id']}: {result}")
        if result.startswith("Error:") or '"error"' in result:
            failed += 1
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
