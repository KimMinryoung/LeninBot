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
    "국가가 자본가를 대신해 자본가 노릇을 하는 체제를 가리키는 말이다. 다만 마르크스주의 안에서 "
    "이 말은 서로 반대 방향으로 쓰여 왔다. 엥겔스는 국가가 공장을 소유해도 노동자의 처지는 "
    "그대로이니 국가가 '이상적 총자본가'가 되었을 뿐이라며 비판의 뜻으로 썼고, 레닌은 노동자가 "
    "권력을 쥔 상태에서 자본주의적 방식을 한동안 활용하는 전술이라는 뜻으로 썼다. 토니 클리프 "
    "이후로는 소련을 자본주의로 고발하는 말로 굳어졌다. 어느 쪽이든 판가름의 축은 그 국가가 "
    "누구의 것인가이다."
)

DEFINITION_EN = (
    "A term for a system in which the state does the work of the capitalist. Within "
    "Marxism the phrase has been used in opposite directions. Engels used it as an "
    "accusation: when the state owns the factories, workers are still employees drawing "
    "wages, so the state has not abolished the capitalist but merged all capitalists into "
    "a single 'ideal collective capitalist.' Lenin used it for the reverse case, a tactic "
    "by which workers who already hold power make temporary use of capitalist methods. "
    "From Tony Cliff onwards it hardened into a charge that the USSR was itself "
    "capitalist. Whichever sense is meant, the question that settles it is the same: "
    "whose state is it?"
)

BODY_KO = """## 한 단어, 두 가지 쓰임

'국가자본주의'는 하나의 학설 이름이 아니다. 이 말은 남의 체제를 비판할 때도 쓰이고, 자기편의 전술적 선택을 설명할 때도 쓰인다. 같은 단어가 정반대 방향으로 쓰이는 셈이라, 누가 어느 뜻으로 썼는지를 먼저 가려야 논의가 성립한다.

비판으로 쓰는 쪽의 계보는 엥겔스의 『반뒤링』(1878)에서 시작한다. 국가가 공장과 철도를 소유해도 노동자는 여전히 남에게 고용되어 임금을 받는 처지 그대로이니, 국가는 자본가를 없앤 것이 아니라 자본가 전체를 하나로 합친 '이상적 총자본가'가 되었을 뿐이라는 것이다. 빌헬름 리프크네히트는 1896년 이 논지를 한 문장으로 밀어붙여, 비스마르크식 국가사회주의야말로 실은 국가자본주의라 불러야 한다고 했다. 바쿠닌은 노동자 권력을 세운다는 계획이 결국 관료라는 새로운 지배계급을 낳으리라 경고했다.

전술로 쓰는 쪽의 계보는 레닌의 것이다.

## 레닌의 두 가지 국가자본주의

레닌은 이름이 겹치는 두 개념을 전혀 다른 맥락에서 썼다. 이 둘을 섞어 읽으면 그의 말이 앞뒤가 안 맞는 것처럼 보인다.

하나는 『임박한 파국과 그 대책』(1917)의 국가독점자본주의다. 전쟁을 치르면서 국가가 산업을 직접 틀어쥐게 된 상황을 가리키며, 자본주의가 도달한 마지막 단계라는 뜻이다. 레닌은 이것을 사회주의의 반대말로 보지 않았다. 이미 만들어진 그 거대한 관리 기구를 소수가 아니라 전 인민의 이익에 복무하도록 돌려놓기만 하면 그것이 곧 사회주의라는 것이다.

다른 하나는 [『현물세』](/commulingo/docs/lenin-tax-in-kind)(1921)의 과도기적 국가자본주의이며, 방향이 반대다. 혁명 직후 러시아 경제는 대부분 자기 땅에서 농사지어 남는 것을 내다 파는 소규모 생산이었다. 레닌은 이런 조건에서 국가자본주의를 사회주의와 견주어 모자란다고 탓할 것이 아니라 흩어진 소규모 생산과 견주어야 하며, 그에 비하면 오히려 진전이라고 주장했다. 그가 든 예가 독일이다. 잘 짜인 대규모 계획 생산 조직에서 융커-부르주아 제국주의 국가를 지우고 그 자리에 노동자의 국가를 놓으면 사회주의에 필요한 조건이 다 갖춰진다는 것이다. 결정적인 것은 경제를 어떤 형태로 운영하는가가 아니라 그 형태를 쥔 국가가 누구의 것인가다.

| 구분 | 국가독점자본주의(1917) | 과도기적 국가자본주의(1921) |
|---|---|---|
| 국가권력 | 자본가의 국가 | 노동자의 국가 |
| 역사적 위치 | 자본주의의 마지막 단계 | 사회주의로 가는 과도기 |
| 국가의 역할 | 독점자본을 위한 통제 | 노동계급을 위한 통제 |

마오쩌둥의 신민주주의 국가자본주의(1940)도 이 계열이되, 노동자 단독 권력이 아니라 네 계급의 연합 권력을 전제한다는 점이 다르다.

## 소련을 무엇이라 부를 것인가

혁명 직후부터 소련 안에서 비판이 나왔다. 좌파공산주의자들(오신스키, 먀스니코프)은 공장에 한 사람의 지배인을 앉히고 미국식 작업 관리 기법을 들여오는 것을 자본주의로의 회귀라고 공격했고, 보르디가는 소련을 그냥 자본주의로 분석했으며 오토 륄레와 판네쿠크가 1920~30년대에 이론을 다듬었다.

이를 완성된 체계로 만든 것이 토니 클리프의 『러시아의 국가자본주의』(1948년 배포, 1955년 개정)다. 그의 논증은 순서가 뒤집혀 있어 눈여겨볼 만하다. 클리프는 소련에 자본주의의 통상적 표지가 없다는 점을 스스로 인정한다. 시장 가격이 무엇을 얼마나 만들지 정하지도 않고, 노동력이 사고팔리는 상품도 아니며, 물건이 남아돌아 터지는 공황도 없다. 그런데도 자본주의라고 부른 근거는 서방과의 군비경쟁이었다. 기업이 시장에서 경쟁하듯 소련은 국가 단위로 경쟁했고 그 압력이 자본축적과 같은 역할을 했다는 것이다. 분기점은 1928년 제1차 5개년 계획으로 잡았다. 관료가 곧 집단적 자본가 계급이라면, 소련과 그 계열 국가를 제국주의 침략에서 방어할 이유도 없다는 정치적 결론이 따라온다. 앞서 막스 섁트먼은 1940년에 같은 관료를 두고 자본주의도 사회주의도 아닌 제3의 체제, 곧 관료적 집산주의라는 규정을 내놓은 바 있다.

트로츠키는 『배반당한 혁명』(1936)에서 이를 거부했다. 그는 레닌이 쓴 "부르주아지 없는 부르주아 국가"라는 표현을 끌어와, 소비에트 국가가 두 얼굴을 지닌다고 보았다. 공장을 국가 소유로 지켜내는 한에서는 사회주의적이고, 특권층의 더 나은 몫을 힘으로 유지하는 한에서는 부르주아적이라는 것이다. 관료는 권력을 쥐었으되 자식에게 물려줄 재산도 내다 팔 주식도 없으므로 아직 계급이 아니다. 여기서 유명한 구분이 나온다. 소유 제도까지 갈아엎는 사회혁명이 필요한 것이 아니라, 그 위에 올라앉은 정치 권력만 갈아치우는 정치혁명이면 된다는 것이다.

## 왜 같은 나라를 두고 결론이 갈리는가

같은 소련을 보고 정반대 판정이 나오는 이유는 자료가 달라서가 아니라, 다음 다섯 가지 중 무엇을 결정적 기준으로 삼는지가 달라서다.

| 기준 | 묻는 것 |
|---|---|
| 소유 형태 | 공장과 토지가 개인의 것인가 국가의 것인가 |
| 잉여 동원 | 무엇을 만들지 시장이 정하는가 국가 계획이 정하는가 |
| 계급 권력 | 국가를 실제로 움직이는 것이 누구인가 |
| 세계체제 위치 | 남을 수탈하는가, 제국주의에 맞서는가, 종속되어 있는가 |
| 노동자 민주주의 | 일하는 사람들이 실제로 결정에 참여하는가 |

무엇을 앞세우느냐가 결론을 정한다. 소유 형태를 앞세우면 트로츠키의 변질된 노동자국가론이 되어 소련을 방어할 의무가 남고, 노동자 민주주의를 앞세우고 소유를 부차적인 것으로 돌리면 클리프론이 되어 그 의무가 사라진다. 계급 권력을 앞세우되 소유 형태도 인정하면 레닌의 과도기론이 된다.

## 한국에서의 쓰임

한국에서 이 말은 학술 용어이기 전에 정파를 가르는 표지로 쓰인다. 클리프의 이론은 영국 사회주의노동자당과 국제사회주의경향의 토대이고, 한국에서는 노동자연대가 그 흐름을 잇는다.

반대편의 국제볼셰비키경향은, 사적 소유를 없앤 사회를 그래도 자본주의라 부른다면 소유 제도를 기준으로 사회를 판별하는 마르크스의 방법 자체를 버리는 셈이라고 반박한다. 이들은 "소련은 자본주의"라는 진단을 처음 내놓은 사람이 1919년 『테러리즘과 공산주의』의 카우츠키, 곧 볼셰비키를 공격한 사회민주주의자였다는 점도 함께 지적한다. 트로츠키주의 내부에서도 테드 그랜트가 1949년 반박문에서, 클리프의 분석으로는 정작 소련을 방어할 것인가 말 것인가라는 실천적 결론이 나오지 않는다고 짚었다.

한편 이름이 비슷한 국가독점자본주의는 다른 경로로 들어왔다. 박현채는 1985년 신식민지국가독점자본주의론으로, 한국 자본주의를 미국에 대한 종속과 재벌·국가의 융합이라는 두 겹의 규정으로 분석했다. 소련이 아니라 남한을 설명하려는 개념이라는 점에서 앞의 논쟁과는 겨냥하는 대상이 다르다."""

BODY_EN = """## One Phrase, Two Uses

'State capitalism' is not the name of a single doctrine. The phrase is used to condemn somebody else's system, and it is also used to explain a tactical choice of one's own. The same words run in opposite directions, so the first thing to establish in any argument about it is which sense the speaker means.

The condemning lineage begins with Engels's *Anti-Dühring* (1878). When the state owns the factories and the railways, workers are still employed by somebody else for a wage, so the state has not abolished the capitalist; it has merged every capitalist into one 'ideal collective capitalist.' In 1896 Wilhelm Liebknecht pushed the point to its conclusion in a single sentence: Bismarck's state socialism is really state capitalism. Bakunin had warned that a plan to install workers' power would end by producing a new ruling class of officials.

The tactical lineage is Lenin's.

## Lenin's Two State Capitalisms

Lenin used two overlapping names in entirely different contexts, and reading them as one makes his position look self-contradictory.

The first is the state-monopoly capitalism of *The Impending Catastrophe and How to Combat It* (1917). It names what happened when war drove the state to take direct hold of industry, and it means capitalism at its final stage. Lenin did not treat this as the opposite of socialism. Turn that machinery of administration, already built, towards serving the whole people rather than a few, and you have socialism.

The second is the transitional state capitalism of [*The Tax in Kind*](/commulingo/docs/lenin-tax-in-kind) (1921), which runs the other way. The Russian economy after the revolution was mostly small-scale production: households farming their own plot and selling what was left over. Lenin argued that under those conditions state capitalism should not be faulted for falling short of socialism but measured against that scattered small production, next to which it was an advance. His example was Germany. Take a well-organised system of large-scale planned production, strike out the Junker-bourgeois imperialist state, put a workers' state in its place, and the conditions for socialism are all present. What decides the matter is not the form in which the economy is run but whose state runs it.

| | State-monopoly capitalism (1917) | Transitional state capitalism (1921) |
|---|---|---|
| State power | The capitalists' state | The workers' state |
| Historical position | Capitalism's final stage | Transition towards socialism |
| Role of the state | Control on behalf of monopoly capital | Control on behalf of the working class |

Mao's New Democratic state capitalism (1940) belongs to this second lineage, differing in that it presumes power held jointly by four classes rather than by the workers alone.

## What to Call the USSR

Criticism began inside the USSR immediately after the revolution. Left communists such as Osinsky and Myasnikov attacked the installation of a single manager in each factory and the import of American work-management methods as a return to capitalism; Bordiga analysed the USSR simply as capitalism, and Otto Rühle and Pannekoek refined the theory through the 1920s and 1930s.

Tony Cliff turned it into a finished system in *State Capitalism in Russia* (circulated 1948, revised 1955). His argument is worth following because it runs backwards. Cliff himself grants that the usual marks of capitalism are missing in the USSR: market prices did not decide what got produced, labour power was not a commodity bought and sold, and there were no crises of goods piling up unsold. His ground for calling it capitalist anyway was military competition with the West. As firms compete in a market, the USSR competed as a state, and that pressure did the work capital accumulation does elsewhere. He dated the break to the First Five-Year Plan of 1928. If the bureaucracy is itself a collective capitalist class, the political conclusion follows that there is no reason to defend the USSR or states like it against imperialist attack. Max Shachtman had already, in 1940, called the same bureaucracy the ruler of a third system that was neither capitalist nor socialist, which he named bureaucratic collectivism.

Trotsky rejected the designation in *The Revolution Betrayed* (1936). Borrowing Lenin's phrase 'a bourgeois state without a bourgeoisie,' he held that the Soviet state had two faces: socialist insofar as it kept the factories in state hands, bourgeois insofar as it used force to preserve a larger share for a privileged layer. The bureaucracy held power but had no property to leave to its children and no shares to sell, and so was not yet a class. From this comes his well-known distinction: what was needed was not a social revolution overturning the property system, but a political revolution replacing only the political power sitting on top of it.

## Why the Same Country Yields Opposite Verdicts

Opposite verdicts on the same USSR follow not from different evidence but from ranking these five tests differently.

| Test | The question it asks |
|---|---|
| Form of ownership | Do the factories and the land belong to individuals or to the state? |
| Mobilisation of surplus | Does the market decide what gets produced, or does state planning? |
| Class power | Who actually moves the state? |
| Position in the world system | Does it exploit others, resist imperialism, or depend on it? |
| Workers' democracy | Do the people who do the work take part in the decisions? |

Put ownership first and you arrive at Trotsky's degenerated workers' state, where the duty to defend the USSR survives; put workers' democracy first and demote ownership and you arrive at Cliff, where it does not; put class power first while still granting weight to ownership and you arrive at Lenin's transitional account.

## The Term in South Korea

In South Korea the phrase works as a marker of political current before it works as an analytical term. Cliff's theory is the foundation of the British Socialist Workers Party and the International Socialist Tendency, and Workers' Solidarity (노동자연대) continues that line.

Against them, the International Bolshevik Tendency argues that to call a society capitalist after private ownership has been abolished is to discard Marx's own method of judging a society by its property system. They add that the first author of the diagnosis 'the USSR is capitalist' was Kautsky, the social democrat attacking the Bolsheviks, in *Terrorism and Communism* (1919). Within Trotskyism, Ted Grant's 1949 reply objected that no practical conclusion about whether to defend the USSR follows from Cliff's analysis at all.

The similarly named state monopoly capitalism reached Korea by a separate route. In 1985 Park Hyun-chae advanced neo-colonial state monopoly capitalism, analysing South Korean capitalism through a double determination: dependence on the United States, and the fusion of the chaebol with the state. Its target is South Korea rather than the USSR, which sets it apart from the debate above."""

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
    "https://www.marxists.org/archive/liebknecht-w/1896/08/our-congress.htm — Wilhelm Liebknecht, 'Our Recent Congress', 10 August 1896: 'Nobody has combatted State Socialism more than we German Socialists, nobody has shown more distinctively than I, that State Socialism is really State capitalism!' The source of the 1896 line, and confirmation that it is Wilhelm rather than his son Karl",
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
    # Not 'liebknecht': that record is Karl, the Spartacist. The 1896 line is his
    # father Wilhelm, who is not in the people dictionary, so he is named in prose
    # rather than linked to the wrong man.
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
