#!/usr/bin/env python3
"""One-off: enrich the 'state-capitalism' CommuLingo glossary term.

The entry described state capitalism only as a critique concept, running
Engels to Cliff, and never mentioned that Lenin used the same phrase in the
opposite direction for a transitional tactic under proletarian power. That
omission is what makes the term read as a single doctrine rather than a
contested word. This rewrite adds Lenin's two distinct concepts, the substance
of the Cliff/Trotsky dispute, the five criteria the dispute actually turns on,
and the Korean reception.

Every factual claim here was checked against the cited primary text, not the
internal review report: the report misdated Ted Grant's reply (1949, not 1948),
compressed Cliff's publication history, and attributed a Lenin quotation to a
page that does not contain it. Those are corrected below.

Run through the systemd credential wrapper:
    sudo bash scripts/run_enrich_state_capitalism.sh --apply
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TERM_ID = "state-capitalism"

DEFINITION_KO = (
    "국가가 자본가 계급의 기능을 수행하는 체제를 가리키는 말이지만, 마르크스주의 안에서 "
    "서로 양립하지 않는 여러 용법으로 쓰여 왔다. 엥겔스는 국가 소유가 자본주의를 폐지하지 "
    "않고 국가를 '이상적 총자본가'로 만들 뿐이라고 보았고, 레닌은 반대로 프롤레타리아 국가가 "
    "자본주의적 형태를 통제하는 과도기 전술이라는 뜻으로 썼다. 토니 클리프 이후 좌파공산주의와 "
    "반체제 트로츠키주의 진영은 이를 소련의 계급 성격을 비판하는 틀로 전환했다. 어느 용법이든 "
    "판가름의 축은 국가권력이 어느 계급의 손에 있는가이다."
)

DEFINITION_EN = (
    "A term for a system in which the state performs the functions of the capitalist "
    "class, used in the Marxist tradition in several mutually incompatible senses. "
    "Engels held that state ownership does not abolish capitalism but merely turns the "
    "state into the 'ideal collective capitalist'; Lenin used the same phrase in the "
    "opposite direction, for a transitional tactic by which a proletarian state controls "
    "capitalist forms; and from Tony Cliff onwards, left communists and dissident "
    "Trotskyists turned it into a framework for indicting the class character of the "
    "USSR. Across every usage the dividing question is the same: which class holds "
    "state power."
)

BODY_KO = """## 두 갈래의 계보

비판 개념 쪽은 엥겔스의 『반뒤링』(1878)에서 시작한다. 국가가 생산수단을 소유해도 노동자는 임금노동자로 남고 국가는 '이상적 총자본가'가 될 뿐이라는 것이다. 리프크네히트는 1896년 비스마르크의 국가사회주의를 '최악의 형태의 자본주의'라 불렀고, 바쿠닌은 프롤레타리아 독재가 관료라는 새로운 계급의 지배로 귀결되리라 경고했다. 다른 하나는 전술 개념이고, 그쪽 계보는 레닌의 것이다.

## 레닌의 두 가지 국가자본주의

레닌은 이름이 겹치는 두 개념을 전혀 다른 맥락에서 썼다. 『임박한 파국과 그 대책』(1917)의 국가독점자본주의는 독점자본이 국가 장치와 융합한 자본주의의 최고 단계이며, 사회주의란 이 독점을 전 인민의 이익에 복무하게 만든 한 걸음에 불과하다는 정식이 여기서 나온다.

『현물세』(1921)의 과도기적 국가자본주의는 방향이 반대다. 레닌은 러시아 경제를 가부장적 농민경제, 소상품생산, 사적 자본주의, 국가자본주의, 사회주의의 다섯 형태로 나누고, 소상품생산이 지배적인 조건에서 국가자본주의는 사회주의가 아니라 소생산자 요소와 견주어야 하며 그에 비하면 한 걸음 전진이라고 주장했다. 그가 든 예가 독일이다. 같은 대규모 계획적 생산조직에서 융커-부르주아 제국주의 국가를 지우고 프롤레타리아 국가를 대신 놓으면 사회주의에 필요한 조건의 총계가 된다. 결정적인 것은 경제 형태가 아니라 그 형태를 쥔 국가의 계급성이다.

| 구분 | 국가독점자본주의(1917) | 과도기적 국가자본주의(1921) |
|---|---|---|
| 국가권력 | 부르주아 국가 | 프롤레타리아 국가 |
| 역사적 위치 | 자본주의 최고 단계 | 사회주의로 가는 과도기 |
| 국가의 역할 | 독점자본을 위한 통제 | 노동계급을 위한 통제 |

마오쩌둥의 신민주주의 국가자본주의(1940)도 이 계열이되, 프롤레타리아 독재가 아니라 네 계급 연합독재를 전제한다는 점이 다르다.

## 소련 성격 논쟁

혁명 직후 좌파공산주의자들(오신스키, 먀스니코프)은 일인관리제와 테일러주의를 자본주의적 생산관계의 복원으로 비판했고, 보르디가는 소련을 그냥 자본주의로 분석했으며 오토 륄레와 판네쿠크가 1920~30년대에 이론을 다듬었다. 이를 완성된 체계로 만든 것이 토니 클리프의 『러시아의 국가자본주의』(1948년 배포, 1955년 개정)다. 클리프는 소련에 가치법칙의 지배도, 노동력의 상품화도, 과잉생산 공황도 없음을 인정하면서, 서방과의 군비경쟁이 자본축적의 기능적 등가물이라는 논리로 소련을 자본주의로 규정하고 분기점을 1928년 제1차 5개년 계획에 두었다. 관료가 집단적 자본가 계급이라면 소련과 그 계열 국가를 제국주의 침략에서 방어할 의무도 없다는 결론이 따라온다. 앞서 막스 섁트먼은 1940년에 같은 관료를 두고 자본주의도 사회주의도 아닌 제3의 체제, 곧 관료적 집산주의라는 규정을 내놓은 바 있다.

트로츠키는 『배반당한 혁명』(1936)에서 이를 거부했다. 그는 레닌이 쓴 "부르주아지 없는 부르주아 국가"라는 표현을 끌어와, 소비에트 국가가 국유화를 지키는 한 사회주의적이고 소비의 불평등을 강제하는 한 부르주아적인 이중 기능을 지닌다고 보았다. 관료는 권력을 행사하되 고유한 소유 기반이 없으므로 아직 계급이 아니다. 그래서 필요한 것은 소유관계를 바꾸는 사회혁명이 아니라 정치적 상부구조만 갈아치우는 정치혁명이라는 결론이 나온다.

## 판별 기준과 한국에서의 수용

논쟁은 결국 다음 다섯 기준 중 무엇을 앞세우느냐로 갈린다.

| 기준 | 내용 |
|---|---|
| 소유 형태 | 생산수단이 사적 소유인가 국가 소유인가 |
| 잉여 동원 | 시장과 이윤인가 계획과 행정명령인가 |
| 계급 권력 | 누가 국가권력을 실제로 행사하는가 |
| 세계체제 위치 | 제국주의적인가, 반제국주의적인가, 종속적인가 |
| 노동자 민주주의 | 생산자의 민주적 통제가 실재하는가 |

소유 형태를 앞세우면 트로츠키의 변질된 노동자국가론이 되어 방어 의무가 남고, 노동자 민주주의를 앞세우고 소유를 부차화하면 클리프론이 되어 방어 의무가 사라진다. 계급 권력을 앞세우되 소유 형태를 인정하면 레닌의 과도기론이 된다.

한국에서 이 용어는 정파와 강하게 결부되어 있다. 클리프의 이론은 영국 사회주의노동자당과 국제사회주의경향의 토대이고 노동자연대가 그 흐름을 잇는다. 반대편의 국제볼셰비키경향은 사적 소유가 철폐되었는데도 자본주의라 부르는 것은 소유체제 중심의 유물론을 포기하는 일이며, "소련은 자본주의"라는 진단의 원조는 1919년 『테러리즘과 공산주의』의 카우츠키라고 반박한다. 트로츠키주의 내부에서도 테드 그랜트가 1949년 반박문에서 클리프의 분석으로는 소련을 방어할 것인가라는 실천적 결론이 나오지 않는다고 지적했다. 한편 국가독점자본주의 계보는 따로 들어와, 박현채가 1985년 신식민지국가독점자본주의론으로 한국 자본주의를 대미 종속과 재벌·국가 융합의 이중 규정으로 분석했다."""

BODY_EN = """## Two Lineages

The critical lineage begins with Engels's *Anti-Dühring* (1878): when the state owns the means of production, workers remain wage-labourers and the state merely becomes the 'ideal collective capitalist.' In 1896 Liebknecht called Bismarck's state socialism 'the worst form of capitalism,' and Bakunin had warned that the dictatorship of the proletariat would end in rule by a new class of bureaucrats. The other lineage is tactical, and it is Lenin's.

## Lenin's Two State Capitalisms

Lenin used two overlapping names in entirely different contexts. The state-monopoly capitalism of *The Impending Catastrophe and How to Combat It* (1917) is the highest stage of capitalism, in which monopoly capital fuses with the state apparatus; socialism, on this account, is no more than the step that makes that monopoly serve the whole people.

The transitional state capitalism of *The Tax in Kind* (1921) runs the other way. Lenin divided the Russian economy into five forms (patriarchal peasant farming, small commodity production, private capitalism, state capitalism, and socialism) and argued that where small commodity production predominates, state capitalism must be measured not against socialism but against the small-proprietor element, against which it is a step forward. His example was Germany: take the same large-scale planned organisation of production, strike out the Junker-bourgeois imperialist state and put a proletarian state in its place, and you have the sum total of the conditions necessary for socialism. What decides the character of the form is not the form but the class holding the state.

| | State-monopoly capitalism (1917) | Transitional state capitalism (1921) |
|---|---|---|
| State power | Bourgeois state | Proletarian state |
| Historical position | Highest stage of capitalism | Transition towards socialism |
| Role of the state | Control on behalf of monopoly capital | Control on behalf of the working class |

Mao's New Democratic state capitalism (1940) belongs to this second lineage, differing in that it presumes a joint dictatorship of four classes rather than a dictatorship of the proletariat.

## The Debate Over the Character of the USSR

Immediately after the revolution, left communists such as Osinsky and Myasnikov attacked one-man management and Taylorism as a restoration of capitalist relations of production; Bordiga analysed the USSR simply as capitalism, and Otto Rühle and Pannekoek refined the theory through the 1920s and 1930s. Tony Cliff turned it into a finished system in *State Capitalism in Russia* (circulated 1948, revised 1955). Cliff conceded that the law of value did not govern Soviet production, that labour power was not a commodity there and that no crises of overproduction occurred, yet still classed the USSR as capitalist on the ground that military competition with the West functioned as the equivalent of capital accumulation, dating the break to the First Five-Year Plan of 1928. If the bureaucracy is a collective capitalist class, no duty follows to defend the USSR or states like it against imperialist attack. Max Shachtman had already, in 1940, described the same bureaucracy as ruling a third system that was neither capitalist nor socialist, which he called bureaucratic collectivism.

Trotsky rejected the designation in *The Revolution Betrayed* (1936). Borrowing Lenin's formulation of a 'bourgeois state without a bourgeoisie,' he held that the Soviet state had a dual function: socialist insofar as it defended nationalised property, bourgeois insofar as it enforced inequality in consumption. The bureaucracy wielded power without a property base of its own and so was not yet a class. What was required was therefore not a social revolution altering property relations but a political revolution replacing the political superstructure alone.

## Criteria, and the Korean Reception

The dispute turns on which of five criteria is given priority.

| Criterion | Question |
|---|---|
| Form of ownership | Are the means of production privately or state owned? |
| Mobilisation of surplus | Market and profit, or plan and administrative command? |
| Class power | Which class actually exercises state power? |
| Position in the world system | Imperialist, anti-imperialist, or dependent? |
| Workers' democracy | Do producers exercise real democratic control? |

Put ownership first and you arrive at Trotsky's degenerated workers' state, where the duty of defence survives; put workers' democracy first and subordinate ownership and you arrive at Cliff, where it does not; put class power first while still granting weight to ownership and you arrive at Lenin's transitional account.

In South Korea the term is strongly identified with particular currents. Cliff's theory is the foundation of the British Socialist Workers Party and the International Socialist Tendency, and Workers' Solidarity (노동자연대) continues that line. Against them, the International Bolshevik Tendency argues that calling a society capitalist after private ownership has been abolished abandons the ownership-centred materialism of Marx, and that the original author of the diagnosis 'the USSR is capitalist' was Kautsky in *Terrorism and Communism* (1919). Within Trotskyism, Ted Grant's 1949 reply objected that no practical conclusion about defending the USSR follows from Cliff's analysis at all. The state-monopoly capitalism lineage entered Korea separately: in 1985 Park Hyun-chae advanced neo-colonial state monopoly capitalism, analysing South Korean capitalism through the double determination of dependence on US imperialism and the fusion of the chaebol with the state."""

SOURCES = [
    "https://en.wikipedia.org/wiki/State_capitalism — comprehensive overview: Engels's formulation of the state as 'ideal collective capitalist,' Bakunin's early critique, left-communist and Trotskyist analyses of the USSR, Bordiga's characterization of the Soviet Union as capitalist, and Maoist adoption of the term",
    "https://ru.wikipedia.org/wiki/Государственный_капитализм — Russian article confirming Lenin's positive use of the term in 1918/1921, Bukharin's pre-1917 analysis, Bakunin's prescient critique of the 'new class,' and contemporary Russian usage",
    "https://www.marxists.org/archive/lenin/works/1921/apr/21.htm — Lenin, 'The Tax in Kind' (April 1921): the five forms of economy in Russia, the argument that state capitalism must be compared with the small-proprietor element rather than with socialism and is a step forward against it, the Germany example showing that the class character of the state and not the economic form is decisive, and concessions and co-operatives as variants of state capitalism",
    "https://www.marxists.org/archive/lenin/works/1917/ichtci/ — Lenin, 'The Impending Catastrophe and How to Combat It' (1917): state-monopoly capitalism as the material threshold of socialism, and the insistence that whose state it is decides what the same economic machinery means",
    "https://www.marxists.org/archive/cliff/works/1955/statecap/ — Tony Cliff, 'State Capitalism in Russia': the 1988 introduction gives the publication history (written 1947, circulated June 1948 as 'The Nature of Stalinist Russia,' revised 1955, retitled by Pluto in 1974) and the appendix directed against Shachtman's bureaucratic collectivism; the text argues the bureaucracy is a collective capitalist class and arms competition the motor of accumulation",
    "https://www.marxists.org/archive/trotsky/1936/revbet/ch03.htm — Trotsky, 'The Revolution Betrayed,' ch. 3: the 'bourgeois state without a bourgeoisie' formulation taken over from Lenin, the dual function of the Soviet state, and the bureaucracy as a privileged stratum without a property base of its own",
    "https://www.marxists.org/archive/grant/1949/cliff.htm — Ted Grant, 'Against the Theory of State Capitalism: Reply to Comrade Cliff' (1949), the orthodox Trotskyist rebuttal arguing that no practical conclusion on defence follows from Cliff's analysis",
    "https://www.marxists.org/reference/archive/mao/selected-works/volume-2/mswv2_26.htm — Mao Zedong, 'On New Democracy' (1940): state capitalism as a component of the New Democratic economy under the joint dictatorship of four classes",
    "https://bolky.jinbo.net — 국제볼셰비키경향(볼셰비키그룹), the Korean Trotskyist organisation whose journal '1917' carries the sustained critique of Cliff: ownership-form materialism, the claim that Kautsky's 'Terrorism and Communism' (1919) is the origin of the 'USSR is capitalist' diagnosis, and the argument that Cliff's dating of 1928 places him closer to Bukharin's Right Opposition than to Trotsky",
    "https://cyber-lenin.com/reports/research/state-capitalism-theory-review — internal review report (2026-07-25) collating the usages, the five discriminating criteria, and the Korean reception; its bibliographic details were re-checked against the primary texts above and corrected where they diverged",
]

PEOPLE = [
    "lenin",
    "trotsky",
    "amadeo-bordiga",
    "anton-pannekoek",
    "mikhail-bakunin",
    "karl-kautsky",
    "mao-zedong",
    "bukharin",
    "liebknecht",
    "osinsky",
    "raya-dunayevskaya",
]

EVENTS = ["new-economic-policy"]


def build_patch() -> dict:
    return {
        "definition": {"ko": DEFINITION_KO, "en": DEFINITION_EN},
        "body": {"ko": BODY_KO, "en": BODY_EN},
        "people": PEOPLE,
        "events": EVENTS,
        "sources": SOURCES,
    }


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true",
                        help="write the change; otherwise print a dry-run summary")
    args = parser.parse_args()

    patch = build_patch()
    print(f"term: {TERM_ID}")
    print(f"  definition ko {len(DEFINITION_KO)} chars (limit 400)")
    print(f"  definition en {len(DEFINITION_EN)} chars (limit 900)")
    print(f"  body ko       {len(BODY_KO)} chars")
    print(f"  body en       {len(BODY_EN)} chars")
    print(f"  sources       {len(SOURCES)}")
    print(f"  people        {len(PEOPLE)} -> {', '.join(PEOPLE)}")
    print(f"  events        {', '.join(EVENTS)}")

    if len(DEFINITION_KO) > 400 or len(DEFINITION_EN) > 900:
        print("ABORT: definition exceeds the card limit", file=sys.stderr)
        return 1

    if not args.apply:
        print("\ndry run; pass --apply to write")
        return 0

    from runtime_tools.commulingo_people import _exec_commulingo_write

    result = await _exec_commulingo_write(
        "term", "update", TERM_ID, SOURCES, patch, 0.95,
    )
    print("\n" + result)
    return 0 if not result.startswith("Error:") else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
