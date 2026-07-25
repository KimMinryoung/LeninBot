#!/usr/bin/env python3
"""Register the 'state-socialism' CommuLingo glossary term.

The glossary had no entry for 국가사회주의 even though it is the sibling of
state-capitalism and the more heavily contaminated of the two words in Korean.

The four usages below were reorganised against the primary texts rather than
taken as received. A prior internal sketch had listed a "traditional, neutral to
positive Marxist-Leninist usage" attributed to Engels's Anti-Dühring and Lenin's
State and Revolution. That is backwards in both cases: Anti-Dühring never uses
the phrase and is dismissive of state ownership as socialism, and the single
occurrence in State and Revolution ch. 4 calls the label an "erroneous bourgeois
reformist assertion". Marx likewise attacks the Lassallean programme in the
Gotha Critique without using the term. So the classics supply no approving
usage at all, and the entry says so.

The 1896 Liebknecht line belongs to Wilhelm, Marx's associate, not to his son
Karl, who is the Liebknecht in the people dictionary. His actual formulation is
sharper than the one previously in circulation here: not that Bismarck's system
was 'the worst form of capitalism' but that state socialism ought to be called
state capitalism outright.

Usage:
  bash scripts/run_register_state_socialism.sh            # dry run
  bash scripts/run_register_state_socialism.sh --apply
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TERM_ID = "state-socialism"

DEFINITION_KO = (
    "국가가 경제를 장악해 사회주의를 실현한다는 구상, 또는 그렇게 운영되는 체제를 가리키는 말이다. "
    "1860년대 라살레가 국가를 사회주의 실현의 도구로 본 데서 비롯해, 1880년대 비스마르크의 사회보험과 "
    "국유화를 부르는 이름이 되었다. 마르크스주의 전통에서는 자기를 가리키는 말로 쓰인 적이 없고 남의 "
    "오류를 지적할 때만 쓰였다. 마르크스·엥겔스·레닌이 모두 이를 공격했고, 빌헬름 리프크네히트는 "
    "1896년 이것을 국가자본주의라 불러야 한다고 했다. 한국에서는 나치의 Nationalsozialismus를 "
    "민족사회주의로 갈라 부르면서 이 말의 자리가 정리되어 왔고, 오늘날 한국어에서 국가사회주의는 "
    "주로 소련형 체제를 가리킨다."
)

DEFINITION_EN = (
    "The idea that socialism is achieved by the state taking hold of the economy, or a "
    "system run that way. It descends from Lassalle, who in the 1860s treated the state "
    "as the instrument of working-class emancipation, and became the name for Bismarck's "
    "social insurance and nationalisations in the 1880s. In the Marxist tradition it has "
    "never been a self-description, only a charge levelled at others: Marx, Engels and "
    "Lenin all attacked it, and in 1896 Wilhelm Liebknecht said it ought to be called "
    "state capitalism instead. Korean usage has been settling the Nazis' quite separate "
    "Nationalsozialismus onto 민족사회주의, so that 국가사회주의 now points mainly at "
    "Soviet-type systems."
)

BODY_KO = """## 누가 쓰느냐에 따라 뜻이 뒤집히는 말

'국가사회주의'는 스스로를 부르는 이름으로 쓰인 적이 거의 없다. 대개는 남을 가리킬 때 쓴다. 국가가 나서서 경제를 장악하면 그것이 곧 사회주의라는 생각이 말 안에 들어 있는데, 바로 그 생각이 마르크스주의가 가장 집요하게 공격한 표적이었다. 한국어에서는 여기에 번역 문제까지 겹친다. 아래 네 가지 쓰임을 갈라 보면 정리된다.

## 1. 라살레와 비스마르크: 말이 생긴 자리

원래 독일어는 Staatssozialismus다. 페르디난트 라살레는 1860년대에 국가를 노동자 해방의 도구로 보았다. 기존 국가를 부수는 대신 그 힘을 빌려 생산자 협동조합을 키우면 사회주의에 이를 수 있다는 구상이었다.

이 발상은 엉뚱한 사람의 손에서 실현됐다. 비스마르크는 1883년부터 1889년까지 의료·재해·노령 보험을 도입하고 주요 철도를 국유화했다. 사회주의자탄압법으로 사민당을 불법화해 놓고 동시에 사민당의 요구를 국가가 가져다 쓴 것이다. 자유주의 진영과 보수 진영은 이를 비꼬아 '국가사회주의'라 불렀고, 비스마르크 자신은 자기가 사민당 사람들보다 더 실천적인 사회주의자라고 응수했다.

## 2. 마르크스주의가 공격한 표적

흔한 오해부터 짚어야 한다. 마르크스·엥겔스·레닌이 이 말을 긍정적인 뜻으로 쓴 적은 없다. 셋 다 공격 대상으로만 언급했다.

마르크스는 『고타강령 비판』(1875)에서 라살레파가 국가 보조금으로 협동조합을 세우자고 한 대목을 겨냥해, 국가 융자로 새 사회를 지을 수 있다면 철도도 그렇게 지으면 되겠다고 비꼬았다. 협동조합은 노동자 자신의 독립적 창조물일 때에만 가치가 있지 정부가 돌보는 피후견물이어서는 안 된다는 것이다.

엥겔스는 더 직설적이다. 『공상에서 과학으로』(1880)의 각주에서 그는 비스마르크식 국유화를 사회주의라 부르는 풍조를 "사이비 사회주의"라 못박았다. 국가가 담배 전매를 한 것이 사회주의라면 나폴레옹과 메테르니히도 사회주의의 창시자에 넣어야 한다는 것이다. 벨기에가 철도를 깔고 비스마르크가 프로이센 철도를 사들인 것은 전시에 다루기 편하자는 계산이지 사회주의와 무관하며, 그 논리대로라면 왕립 도자기 공장과 군대의 연대 재봉사까지 사회주의 기관이 된다고 조롱했다.

빌헬름 리프크네히트는 1896년에 이 논쟁을 한 문장으로 정리했다. "국가사회주의를 가장 앞장서 반대해 온 것이 바로 우리 독일 사회주의자들이며, 국가사회주의가 실은 국가자본주의라는 것을 누구보다 분명히 보인 사람이 나다." 이 말에 쓸 자리가 있다면 그 자리는 국가자본주의의 것이라는 주장이다.

레닌도 같은 선에 선다. 『국가와 혁명』(1917)에서 그는 독점자본주의나 국가독점자본주의는 이제 자본주의가 아니라 '국가사회주의'라 부를 수 있다는 주장을 "부르주아 개량주의의 그릇된 단언"이라고 잘라 말했다.

## 3. 나치의 민족사회주의와 어떻게 갈리는가

한국에서는 나치를 '민족사회주의', 소련형 체제를 '국가사회주의'로 갈라 부르는 쪽으로 정리되어 왔다. 이 구분은 타당하고, 독일어를 보면 이유가 분명하다. 나치의 이념은 Nationalsozialismus이고 이 항목이 다루는 말은 Staatssozialismus다. 처음부터 다른 단어이며, 앞의 것을 곧이곧대로 옮기면 민족사회주의가 된다.

내용에서도 겹치지 않는다. Staatssozialismus는 국가가 경제를 운영한다는 경제 구상이고, Nationalsozialismus는 계급 대신 민족을 앞세워 마르크스주의의 계급투쟁과 국제주의를 정면으로 부정한 이념이다. 히틀러 자신이 사회주의라는 말을 고대 아리아·게르만의 제도라 주장하며 계급과 무관한 뜻으로 다시 정의했다.

다만 정리가 끝난 것은 아니다. Nationalsozialismus의 번역어로는 민족사회주의 외에 국민사회주의와 국가사회주의가 함께 쓰여 왔고, 나치당의 명칭은 지금도 '국가사회주의 독일 노동자당'으로 표기되는 경우가 많다. 한국어 위키백과의 당 문서 표제가 그렇다. 근래에 민족사회주의로 부르자는 주장이 힘을 얻은 이유가 바로 이 표기가 라살레 계열의 Staatssozialismus와 충돌하기 때문이다. 즉 두 말을 갈라놓으려는 노력 자체가, 한국어에서 '국가사회주의'의 자리를 이 항목이 다루는 계보에 돌려주려는 작업이다.

## 4. 1989년 이후 학계의 분류 이름

동구권이 무너진 뒤 서구 학계는 소련형 체제를 통칭하는 표준 용어로 'state socialism'을 쓰기 시작했다. "국가사회주의는 왜 붕괴했는가" 같은 제목이 이 용법이다.

짚어둘 점이 있다. 소련과 동구 국가들은 스스로를 그렇게 부른 적이 없다. 이 이름은 바깥에서 사후에 붙인 분류이고, 사회주의가 실패했다는 판단을 이미 전제한 채 그 사례들을 묶는 범주로 기능한다. 학술 문헌에서 이 표현을 만나면 서술자가 어디에 서 있는지를 함께 읽어야 하는 이유다.

## 왜 분석 도구가 되지 못했는가

네 가지 쓰임을 겹쳐 놓으면 공통점이 드러난다. 국가가 소유하고 운영하면 그것이 사회주의라는 등식이 말 안에 이미 박혀 있다는 것이다.

마르크스주의의 반론은 일관된다. 국가가 무엇을 소유하는가보다 그 국가가 누구의 것인가를 먼저 물어야 한다는 것이다. '국가사회주의'는 그 질문을 덮어버리기 때문에 분석 도구가 되지 못하고 비판의 대상으로 남았다. 같은 질문을 정면으로 던지려고 마련된 짝이 국가자본주의이며, 빌헬름 리프크네히트가 1896년에 요구한 것이 바로 그 교체였다."""

BODY_EN = """## A Word That Reverses Depending on Who Uses It

'State socialism' has almost never been a name anyone claims for themselves. It is mostly applied to other people. Built into it is the assumption that once the state takes hold of the economy the result is socialism, and that assumption is precisely what the Marxist tradition attacked hardest. In Korean a translation problem is laid on top of this. Separating the four usages below clears it up.

## 1. Lassalle and Bismarck: Where the Word Comes From

The German is Staatssozialismus. In the 1860s Ferdinand Lassalle treated the state as the instrument of working-class emancipation: rather than smashing the existing state, borrow its power to build producers' co-operatives and arrive at socialism that way.

The idea was realised by an unlikely hand. Between 1883 and 1889 Bismarck introduced sickness, accident and old-age insurance and nationalised the main railways, all while the Anti-Socialist Laws kept the Social Democrats banned. Liberal and conservative opponents mockingly called this state socialism, and Bismarck answered that he was a more practical socialist than the Social Democrats were.

## 2. What Marxism Attacked

A common misconception has to be cleared first. Marx, Engels and Lenin never used the phrase approvingly. All three raised it only to attack it.

In the *Critique of the Gotha Programme* (1875) Marx went after the Lassallean demand for state subsidies to co-operatives, remarking that if you can build a new society with state loans then you may as well build a railway that way. Co-operatives are worth something only as the independent creations of the workers, not as wards of the government.

Engels was blunter. In a footnote to *Socialism: Utopian and Scientific* (1880) he called the fashion for describing Bismarckian nationalisation as socialism a 'spurious socialism.' If the state taking over the tobacco trade is socialist, then Napoleon and Metternich must be counted among the founders of socialism. Belgium building its railways and Bismarck buying up the Prussian lines were done to have them in hand in case of war, not out of any socialist purpose, and on that reasoning the royal porcelain works and even the army's regimental tailor would qualify as socialist institutions.

Wilhelm Liebknecht settled the argument in a sentence in 1896: "Nobody has combatted State Socialism more than we German Socialists, nobody has shown more distinctively than I, that State Socialism is really State capitalism!" Whatever work the phrase was doing, he argued, belongs to state capitalism instead.

Lenin stands in the same line. In *The State and the Revolution* (1917) he called the claim that monopoly or state-monopoly capitalism is no longer capitalism but may now be called 'state socialism' an erroneous bourgeois reformist assertion.

## 3. How It Separates From Nazi National Socialism

Korean usage has been settling on 민족사회주의 for the Nazis and 국가사회주의 for Soviet-type systems. The separation is sound, and the German shows why. Nazi ideology was Nationalsozialismus; the word this entry treats is Staatssozialismus. They were never the same word, and the first renders literally as national socialism.

Nor do they overlap in content. Staatssozialismus is an economic proposal about the state running the economy; Nationalsozialismus put nation in place of class and rejected Marxist class struggle and internationalism outright, with Hitler redefining socialism as an ancient Aryan and Germanic institution having nothing to do with class.

The tidying is not finished, though. Nationalsozialismus has been rendered in Korean as 민족사회주의, 국민사회주의 and 국가사회주의 alike, and the party is still commonly written as 국가사회주의 독일 노동자당, which is the title Korean Wikipedia gives its article on the party. The recent push to standardise on 민족사회주의 exists precisely because that rendering collides with Lassalle's Staatssozialismus. The effort to separate the two is, in effect, an effort to return the Korean phrase 국가사회주의 to the lineage this entry describes.

## 4. A Classifying Label in Scholarship After 1989

Once the Eastern bloc collapsed, Western scholarship adopted 'state socialism' as the standard term for Soviet-type systems, as in titles of the form "why did state socialism collapse?"

One thing is worth noting. The USSR and the Eastern bloc states never used it of themselves. The name was applied from outside and after the fact, and it works as a category that groups these cases while already presupposing that socialism failed. That is why the phrase, when met in the academic literature, has to be read together with the position of the writer using it.

## Why It Never Became an Analytical Tool

Lay the four usages over one another and the common element appears: the equation of state ownership and operation with socialism is already lodged inside the word.

The Marxist objection is consistent throughout. The question to ask first is not what the state owns but whose state it is. Because 'state socialism' buries that question, it never became a tool of analysis and remained an object of criticism instead. The counterpart built to put the question directly is state capitalism, and the substitution Wilhelm Liebknecht demanded in 1896 was exactly that."""

SOURCES = [
    "https://ko.wikipedia.org/wiki/국가사회주의 — 한국어 위키백과. Staatssozialismus를 '국가가 경제를 간섭하고 주도하여 사회주의를 실현하려는 사상'으로 정의하고, 나치즘을 뜻하는 민족사회주의(Nationalsozialismus)와 '다른 개념'임을 명시한다. 라살레가 1862년 국가를 사회주의 달성의 도구로 본 것이 사상적 기초이며 비스마르크의 1883~1889년 사회복지제도가 이 이름으로 불렸다는 서술도 여기서 확인된다",
    "https://en.wikipedia.org/wiki/State_socialism — Bismarck's 1883-1889 welfare programmes were 'informally referred to as State Socialism by liberal and conservative opponents'; Bismarck's claim to be a more practical socialist than the Social Democrats; and the note that classical and orthodox Marxists treated the term as an oxymoron",
    "https://www.marxists.org/archive/marx/works/1875/gotha/ch03.htm — Marx, Critique of the Gotha Programme (1875): the attack on the Lassallean demand for state aid to producers' co-operatives, 'with state loans one can build a new society just as well as a new railway', and the insistence that co-operatives count only as the independent creations of the workers. Marx does not use the term 'state socialism' itself here; the target is the programme the label later attached to",
    "https://www.marxists.org/archive/marx/works/1880/soc-utop/ch03.htm — Engels, Socialism: Utopian and Scientific (1880), footnote: Bismarckian state ownership declared socialist is 'a kind of spurious Socialism', with the tobacco monopoly, Napoleon and Metternich, the Belgian railways, the royal porcelain manufacture and the regimental tailor",
    "https://www.marxists.org/archive/liebknecht-w/1896/08/our-congress.htm — Wilhelm Liebknecht, 'Our Recent Congress', Paris, 10 August 1896: 'Nobody has combatted State Socialism more than we German Socialists, nobody has shown more distinctively than I, that State Socialism is really State capitalism!' This is Wilhelm, Marx's associate, not his son Karl",
    "https://www.marxists.org/archive/lenin/works/1917/staterev/ch04.htm — Lenin, The State and Revolution (1917), ch. 4: the only appearance of the phrase in the work, rejecting as an 'erroneous bourgeois reformist assertion' the claim that state-monopoly capitalism is no longer capitalism and may be called 'state socialism'",
    "https://en.wikipedia.org/wiki/Nazism — the party's full name Nationalsozialistische Deutsche Arbeiterpartei, Hitler's redefinition of socialism as 'an ancient Aryan, Germanic institution', and Nazism's rejection of Marxist class conflict and internationalism, which is what separates Nationalsozialismus from Staatssozialismus",
    "https://ko.wikipedia.org/wiki/나치즘 — 한국어 위키백과가 Nationalsozialismus의 번역어로 민족사회주의를 표제로 삼고 국민사회주의·국가사회주의를 병기하는 현황. 한국에서 나치는 민족사회주의, 소련형 체제는 국가사회주의로 갈라 부르는 관행이 자리잡아 왔음을 보여준다",
    "https://ko.wikipedia.org/wiki/국가사회주의_독일_노동자당 — 그 정리가 아직 끝나지 않았다는 증거. 한국어 위키백과의 나치당 문서 표제는 여전히 '국가사회주의 독일 노동자당'이며, 이 표기가 라살레 계열 Staatssozialismus와 충돌하는 것이 민족사회주의로 부르자는 주장의 근거다",
]

PATCH = {
    "term": {"ko": "국가사회주의", "en": "State Socialism"},
    "original": "Staatssozialismus",
    "period": {"ko": "1860년대~현재", "en": "1860s–present"},
    "startYear": 1860,
    "category": "theory",
    "definition": {"ko": DEFINITION_KO, "en": DEFINITION_EN},
    "body": {"ko": BODY_KO, "en": BODY_EN},
    "aliases": {
        "ko": ["국가사회주의", "국가 사회주의"],
        "en": ["state socialism", "state-socialism", "Staatssozialismus"],
    },
    # Wilhelm Liebknecht is not in the people dictionary; the 'liebknecht' record
    # is his son Karl, so he is named in prose without a link rather than linked
    # to the wrong man.
    "people": ["lenin", "karl-kautsky", "eduard-bernstein"],
    "sources": SOURCES,
}


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    print(f"term: {TERM_ID}")
    print(f"  definition ko {len(DEFINITION_KO)} chars (limit 400)")
    print(f"  definition en {len(DEFINITION_EN)} chars (limit 900)")
    print(f"  body ko       {len(BODY_KO)} chars")
    print(f"  body en       {len(BODY_EN)} chars")
    print(f"  sources       {len(SOURCES)}")
    print(f"  people        {', '.join(PATCH['people'])}")

    if len(DEFINITION_KO) > 400 or len(DEFINITION_EN) > 900:
        print("ABORT: definition exceeds the card limit", file=sys.stderr)
        return 1

    if not args.apply:
        print("\ndry run; pass --apply to write")
        return 0

    from runtime_tools.commulingo_people import _exec_commulingo_write

    result = await _exec_commulingo_write(
        "term", "create", TERM_ID, SOURCES, PATCH, 0.95,
    )
    print("\n" + result)
    return 0 if not result.startswith("Error:") else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
