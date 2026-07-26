#!/usr/bin/env python3
"""Register the 『고타강령 비판』 and 『공상에서 과학으로』 glossary entries.

Eight entries in two nests. The glossary already registers documents as
entries (4월 테제, 류틴 강령, 46인 선언, 10월 선언), so the two works are
entries in their own right, and the concepts each work is read for hang under
them by parent_id — the same one-level nesting 대숙청 uses for 예조프시나.

  critique-of-the-gotha-programme   고타강령 비판           (1875 / 1891)
    lower-and-higher-phase-of-communism  공산주의의 낮은 단계와 높은 단계
    undiminished-proceeds-of-labour      노동전수익권
    iron-law-of-wages                    임금철칙
  socialism-utopian-and-scientific  공상에서 과학으로       (1880)
    utopian-socialism                    공상적 사회주의
    scientific-socialism                 과학적 사회주의
    withering-away-of-the-state          국가의 사멸

Everything is written against the primary texts on marxists.org rather than
from the received summaries. Three places where the received version is wrong
or thin, and this set says so:

- The Critique is routinely quoted for 'from each according to his needs' and
  for the dictatorship-of-the-proletariat sentence, and almost never for what
  occupies most of its pages: a demolition of the idea that the state can be
  the builder of socialism. That is the thread these entries follow, because it
  is what makes the document the origin point of the state-socialism entry.
- Engels's 1891 edition was abridged, and he says so in his own foreword —
  sharp personal expressions replaced by dots, some sentences cut for the Press
  Law, softened wordings in square brackets. Entries that treat the printed
  text as the manuscript are skipping that.
- 'Iron law of wages' is usually filed as an economic doctrine Marx disagreed
  with. Marx's actual objection is narrower and sharper: nothing in it is
  Lassalle's except the word 'iron', and if its Malthusian grounding held, the
  law would survive the abolition of wage labour a hundred times over.

The Korean headword is 임금철칙, not the literal 철의 임금법칙 the English name
invites — this entry was registered under the literal form first and corrected.
Korean Wikipedia titles its article 임금의 철칙 and writes 임금철칙 (賃金鐵則)
elsewhere, while a full-text search there for 철의 임금법칙 returns nothing;
노동자의 책 files the term as 임금철칙 and socialist.kr's Korean serialisation
of the Critique uses 임금철칙설 (both cited by the operator — the two sites
refuse automated fetches, so only the Wikipedia evidence was read directly).
All the circulating forms are registered as aliases, so prose written any of
those ways still auto-links here.

Term↔term relations are not writable through this tool surface (see
_TERM_PATCH_KEYS); they go in the companion frontend migration
scripts/migrations/105_commulingo_gotha_utopian_relations.sql.

Usage:
  bash scripts/run_commulingo_register.sh scripts/register_gotha_utopian_terms.py
  bash scripts/run_commulingo_register.sh scripts/register_gotha_utopian_terms.py --apply
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

SRC_GOTHA_INDEX = (
    f"{MIA}/archive/marx/works/1875/gotha/index.htm — Marx/Engels Selected Works: "
    "'Written: April or early May, 1875… First Published: Abridged in the journal "
    "Die Neue Zeit, Bd. 1, No. 18, 1890-91.' The abridgement is the published text, "
    "not the manuscript"
)
SRC_GOTHA_1 = (
    f"{MIA}/archive/marx/works/1875/gotha/ch01.htm — Part I: 'Labor is not the source "
    "of all wealth. Nature is just as much the source of use values'; the list of "
    "deductions from the total social product; 'equal right here is still in principle "
    "– bourgeois right'; and 'only then can the narrow horizon of bourgeois right be "
    "crossed in its entirety and society inscribe on its banners: From each according "
    "to his ability, to each according to his needs!'"
)
SRC_GOTHA_2 = (
    f"{MIA}/archive/marx/works/1875/gotha/ch02.htm — Part II on the iron law of wages: "
    "'It is well known that nothing of the \"iron law of wages\" is Lassalle's except "
    "the word \"iron\" borrowed from Goethe's \"great, eternal iron laws\"'; the "
    "Malthusian grounding by way of Lange; and the correction that wages are 'only a "
    "masked form for the value, or price, of labor power'"
)
SRC_GOTHA_3 = (
    f"{MIA}/archive/marx/works/1875/gotha/ch03.htm — Part III on state-aided "
    "co-operatives: 'It is worthy of Lassalle's imagination that with state loans one "
    "can build a new society just as well as a new railway!' and 'they are of value "
    "only insofar as they are the independent creations of the workers and not "
    "protégés either of the governments or of the bourgeois'"
)
SRC_GOTHA_4 = (
    f"{MIA}/archive/marx/works/1875/gotha/ch04.htm — Part IV on the 'free state': "
    "'Freedom consists in converting the state from an organ superimposed upon society "
    "into one completely subordinate to it', and 'Between capitalist and communist "
    "society there lies the period of the revolutionary transformation of the one into "
    "the other. Corresponding to this is also a political transition period in which "
    "the state can be nothing but the revolutionary dictatorship of the proletariat.'"
)
SRC_GOTHA_FOREWORD = (
    f"{MIA}/archive/marx/works/1875/gotha/foreword.htm — Engels's foreword, London, "
    "6 January 1891: the manuscript 'was sent in 1875, shortly before the Gotha Unity "
    "Congress, to Bracke for communication to Geib, Auer, Bebel, and Liebknecht and "
    "subsequent return to Marx'; the Halle Congress put the Gotha Programme on the "
    "agenda, so withholding it any longer would make him 'guilty of suppression'; and "
    "'I have omitted a few sharp personal expressions and judgments where these were "
    "immaterial, and replaced them by dots'"
)
SRC_SUS_INDEX = (
    f"{MIA}/archive/marx/works/1880/soc-utop/index.htm — publication history in "
    "Engels's own words in the 1892 English preface: three chapters of Anti-Dühring "
    "arranged as a pamphlet at Paul Lafargue's request, translated and published by "
    "him in 1880; Polish and Spanish from the French; the German original in 1883; "
    "Italian, Russian, Danish, Dutch and Romanian from the German; 'I am not aware "
    "that any other Socialist work, not even our Communist Manifesto of 1848, or "
    "Marx's Capital, has been so often translated.' First published in the March, "
    "April and May 1880 issues of Revue Socialiste"
)
SRC_SUS_1 = (
    f"{MIA}/archive/marx/works/1880/soc-utop/ch01.htm — Part I on the three great "
    "Utopians: Saint-Simon's antagonism of 'workers' and 'idlers' and his forecast of "
    "the absorption of politics by economics; Fourier as the first to declare that the "
    "degree of woman's emancipation is the natural measure of general emancipation; "
    "Owen at New Lanark; and Engels's own verdict — 'we delight in the stupendously "
    "grand thoughts and germs of thought that everywhere break out through their "
    "phantastic covering'"
)
SRC_SUS_2 = (
    f"{MIA}/archive/marx/works/1880/soc-utop/ch02.htm — 'These two great discoveries, "
    "the materialistic conception of history and the revelation of the secret of "
    "capitalistic production through surplus-value, we owe to Marx. With these "
    "discoveries, Socialism became a science.'"
)
SRC_SUS_3 = (
    f"{MIA}/archive/marx/works/1880/soc-utop/ch03.htm — Part III: 'The modern state, "
    "no matter what its form, is essentially a capitalist machine — the state of the "
    "capitalists, the ideal personification of the total national capital'; the "
    "footnote on Bismarckian nationalisation as 'a kind of spurious Socialism' with "
    "the tobacco monopoly, Napoleon and Metternich, the Belgian railways, the royal "
    "porcelain manufacture and the regimental tailor; 'the government of persons is "
    "replaced by the administration of things'; 'The State is not \"abolished\". It "
    "dies out.'; and 'It is the ascent of man from the kingdom of necessity to the "
    "kingdom of freedom.'"
)
SRC_LENIN_1 = (
    f"{MIA}/archive/lenin/works/1917/staterev/ch01.htm — Lenin's correction: 'the "
    "bourgeois state does not \"wither away\", but is \"abolished\" by the proletariat "
    "in the course of the revolution. What withers away after this revolution is the "
    "proletarian state or semi-state', plus the reading of 'free people's state' as an "
    "opportunist catchword"
)
SRC_LENIN_5 = (
    f"{MIA}/archive/lenin/works/1917/staterev/ch05.htm — chapter 5, the commentary on "
    "the Critique: 'What is usually called socialism was termed by Marx the \"first\", "
    "or lower, phase of communist society'; 'under communism there remains for a time "
    "not only bourgeois law, but even the bourgeois state, without the bourgeoisie!'; "
    "and the reading of the higher phase"
)
SRC_STALIN_18TH = (
    f"{MIA}/reference/archive/stalin/works/1939/03/10.htm — Report to the 18th Congress, "
    "10 March 1939, 'Some Questions of Theory': the state will be retained under "
    "communism 'unless the capitalist encirclement is liquidated', and Engels's formula "
    "is said to presuppose socialism victorious in all or most countries — a case the "
    "classics 'could not have given an answer' to"
)
SRC_CONSTITUTION_1936 = (
    f"{MIA}/reference/archive/stalin/works/1936/12/05.htm — Constitution of the USSR "
    "(1936), Article 12: 'In the U.S.S.R. work is a duty and a matter of honour for "
    "every able-bodied citizen, in accordance with the principle: \"He who does not "
    "work, neither shall he eat.\" The principle applied in the U.S.S.R. is that of "
    "socialism: \"From each according to his ability, to each according to his work.\"'"
)
SRC_KO_IRON_LAW_NAME = (
    "https://ko.wikipedia.org/wiki/임금의_철칙 — 한국어 위키백과의 표제어는 '임금의 "
    "철칙'(Ehernes Lohngesetz)이며 라살이 처음 명명했다고 서술한다. '생존권' 문서는 "
    "'임금철칙(賃金鐵則)', '페르디난트 라살' 문서는 '임금의 철칙'으로 쓰고, 위키백과 전문 "
    "검색에서 '철의 임금법칙'은 0건이다"
)
SRC_KO_IRON_LAW_DICT = (
    "https://www.laborsbook.org/dic/view.php?dic_part=dic03&idx=536 — 노동자의 책 "
    "용어사전의 '임금철칙' 항목"
)
SRC_KO_IRON_LAW_GOTHA = (
    "http://socialist.kr/classics-of-socialism-critique-of-the-gotha-program-6/ — "
    "『고타강령 비판』 한국어 연재 6부. 이 교리를 '임금철칙설'로 옮긴다"
)
SRC_LASSALLE = (
    "https://en.wikipedia.org/wiki/Ferdinand_Lassalle — the General German Workers' "
    "Association founded at Leipzig on 23 May 1863 with Lassalle as president for five "
    "years; state-financed producers' co-operatives to be won through universal "
    "suffrage; the secret meetings with Bismarck from May 1863; death on 31 August 1864 "
    "of a duelling wound"
)
SRC_IRON_LAW = (
    "https://en.wikipedia.org/wiki/Iron_law_of_wages — ehernes Lohngesetz; Lassalle's "
    "phrase 'das eiserne und grausame Gesetz'; the descent from Turgot, Ricardo's rent "
    "theory and Malthus's subsistence doctrine; and 'iron' taken from Goethe's 'great, "
    "eternal iron laws' in Das Göttliche"
)
SRC_W_LIEBKNECHT = (
    f"{MIA}/archive/liebknecht-w/1896/08/our-congress.htm — Wilhelm Liebknecht, 'Our "
    "Recent Congress' (1896): 'Nobody has combatted State Socialism more than we German "
    "Socialists, nobody has shown more distinctively than I, that State Socialism is "
    "really State capitalism!'"
)
SRC_GOTHA_CONGRESS = (
    "https://en.wikipedia.org/wiki/Critique_of_the_Gotha_Programme — the May 1875 Gotha "
    "congress at which the Eisenacher Social Democratic Workers' Party and the "
    "Lassallean General German Workers' Association merged into the Socialist Workers' "
    "Party of Germany, adopting the programme with minor changes despite Marx's "
    "objections; Engels published the text in Die Neue Zeit in 1891 over the leadership's "
    "attempt to suppress it"
)
SRC_MENGER = (
    "https://en.wikipedia.org/wiki/Anton_Menger — Das Recht auf den vollen Arbeitsertrag "
    "in geschichtlicher Darstellung (1886), the history of the claim to the whole "
    "produce of labour that fixed the Korean rendering 노동전수익권"
)
SRC_ANTI_DUHRING = (
    f"{MIA}/archive/marx/works/1877/anti-duhring/index.htm — Herr Eugen Dühring's "
    "Revolution in Science (1878), the book the pamphlet's three chapters come from, "
    "written because Dühring's system was winning adherents inside the German party"
)
SRC_SCIENTIFIC_COMMUNISM = (
    "https://en.wikipedia.org/wiki/Scientific_communism — 'scientific communism' made a "
    "compulsory subject in Soviet higher education by order no. 214 of 27 June 1963, on "
    "Mikhail Suslov's initiative, absorbing the sociological part of historical "
    "materialism"
)

# ── 1. 고타강령 비판 ──────────────────────────────────────────────────

GOTHA = {
    "id": "critique-of-the-gotha-programme",
    "sources": [
        SRC_GOTHA_INDEX, SRC_GOTHA_1, SRC_GOTHA_2, SRC_GOTHA_3, SRC_GOTHA_4,
        SRC_GOTHA_FOREWORD, SRC_GOTHA_CONGRESS, SRC_LENIN_5, SRC_CONSTITUTION_1936,
    ],
    "patch": {
        "term": {"ko": "고타강령 비판", "en": "Critique of the Gotha Programme"},
        "original": "Kritik des Gothaer Programms",
        "period": {"ko": "1875년 집필 · 1891년 발표", "en": "Written 1875, published 1891"},
        "startYear": 1875,
        "endYear": 1891,
        "category": "theory",
        "aliases": {
            "ko": ["고타강령 비판", "고타 강령 비판", "고타강령비판"],
            "en": [
                "Critique of the Gotha Programme", "Critique of the Gotha Program",
                "Kritik des Gothaer Programms", "Gotha Critique",
            ],
        },
        "people": [
            "karl-marx", "friedrich-engels", "ferdinand-lassalle",
            "wilhelm-liebknecht", "lenin", "karl-kautsky",
        ],
        "definition": {
            "ko": (
                "1875년 마르크스가 독일 두 노동자 정당의 통합강령 초안 옆에 달아 보낸 방주(旁註). "
                "브라케를 거쳐 지도부만 돌려 읽었고 대회는 거의 무시했다. 원고는 15년 넘게 묻혔다가 "
                "1891년 엥겔스가 축약해 『신시대』에 발표했다. 노동전수익권, 공정한 분배, 임금철칙, "
                "국가 보조 협동조합, 자유국가 — 라살레주의 구호를 차례로 해부하고 그 자리에 "
                "공산주의의 두 단계와 '이 시기의 국가는 프롤레타리아트의 혁명적 독재 외의 다른 것일 수 "
                "없다'는 문장을 놓았다. 국가를 사회주의의 도구로 보는 발상에 대한 마르크스의 마지막 "
                "답변이다."
            ),
            "en": (
                "Marginal notes Marx wrote in 1875 beside the draft programme on which two German "
                "workers' parties were about to merge. Sent through Bracke, they were read by the "
                "leadership alone and largely ignored by the congress. The manuscript stayed buried "
                "for more than fifteen years until Engels published an abridged text in Die Neue "
                "Zeit in 1891. It takes the Lassallean slogans in turn — the undiminished proceeds "
                "of labour, fair distribution, the iron law of wages, state-aided co-operatives, "
                "the free state — and puts in their place the two phases of communist society and "
                "the sentence that the state of the transition period 'can be nothing but the "
                "revolutionary dictatorship of the proletariat'. It is Marx's last answer to the "
                "idea of the state as the instrument of socialism."
            ),
        },
        "body": {
            "ko": """## 축하가 아니라 경고로 쓰인 문서

1875년 5월, 독일의 두 노동자 정당이 고타에서 합당을 앞두고 있었다. 한쪽은 빌헬름 리프크네히트와 아우구스트 베벨의 아이제나흐파(사회민주노동자당)로 마르크스·엥겔스와 가까웠고, 다른 한쪽은 페르디난트 라살레가 1863년에 세운 전독일노동자협회(ADAV)였다. 비스마르크의 제국 아래 흩어진 힘을 모으는 일이었으니 반가운 소식일 법했다. 런던에서 온 것은 축하가 아니었다.

마르크스는 통합강령 초안 옆에 방주를 달았다. 원래 제목은 「독일 노동자당 강령 방주」이고, 『고타강령 비판』은 나중에 붙은 이름이다. 원고는 1875년 4월 말에서 5월 초에 쓰여 빌헬름 브라케에게 보내졌고, 가이프·아우어·베벨·리프크네히트가 돌려 읽은 뒤 마르크스에게 돌아오기로 되어 있었다. 대회는 원고를 사실상 무시했다. 강령은 사소한 수정만 거쳐 채택되었고 새 당은 독일사회주의노동자당이 되었다.

## 네 개의 표적

비판은 강령의 문장을 차례로 뜯어 읽는 방식으로 진행된다. 겨냥한 것은 라살레가 남긴 유산 넷이다.

**노동전수익권.** 강령 첫 조항은 "노동의 수익은 감소되지 않은 채 평등한 권리로 사회의 모든 구성원에게 귀속된다"고 선언한다. 마르크스는 그 앞 구절부터 잘랐다. "노동은 모든 부의 원천이 아니다. 자연도 노동과 꼭 같이 사용가치의 원천이다." 이어 총생산물에서 무엇을 먼저 떼어야 하는지를 열거한다. 소모된 생산수단의 대체분, 확대재생산 몫, 재해에 대비한 예비기금. 남은 소비 몫에서 다시 일반 관리비, 학교와 보건 같은 공동 수요, 노동 능력을 잃은 사람들을 위한 기금. "감소되지 않은" 수익은 이 목록을 지나며 저절로 "감소된" 수익이 된다.

**임금철칙.** 강령은 임금제도를 "임금철칙과 함께" 폐지하겠다고 썼다. 마르크스는 이 법칙에서 라살레의 것은 괴테에게서 빌려 온 '철칙'이라는 낱말뿐이라고 못박았다.

**국가 보조 협동조합.** 강령은 국가의 보조로 생산협동조합을 세워 거기서 사회주의적 노동조직이 "생겨나게" 하자고 요구했다. 대답은 한 문장이다. "국가 융자로 새 철도를 놓듯 새 사회도 지을 수 있다는 것은 라살레의 상상력에나 어울린다." 협동조합은 노동자 자신의 독립적 창조물일 때에만 값이 있지 정부나 부르주아의 피후견물이어서는 안 된다.

**자유국가.** 강령은 "자유로운 국가"를 목표로 걸었다. 마르크스는 되묻는다. 자유국가란 무엇인가. 국가를 자유롭게 하는 것은 노동자의 목표가 아니다. "자유는 국가를 사회 위에 올라앉은 기관에서 사회에 완전히 종속된 기관으로 바꾸는 데 있다."

## 두 단계, 그리고 한 문장

분배 논쟁을 끝까지 밀고 간 끝에 문서는 공산주의 사회를 두 단계로 갈라 보는 대목에 이른다. 자본주의에서 갓 나와 "낡은 사회의 모반(母斑)을 온몸에 지닌" 낮은 단계에서는 생산자가 사회에 준 노동량만큼을 되돌려 받으므로 여전히 부르주아적 권리가 남고, 분업에 대한 예속과 정신노동·육체노동의 대립이 사라진 높은 단계에 이르러서야 사회는 깃발에 새길 수 있다. "각자는 능력에 따라, 각자에게는 필요에 따라!"

4부는 국가 문제를 정면으로 다룬다. "자본주의 사회와 공산주의 사회 사이에는 전자에서 후자로의 혁명적 전화의 시기가 있다. 여기에 대응해 정치적 이행기도 있으며, 이 시기의 국가는 프롤레타리아트의 혁명적 독재 외의 다른 것일 수 없다." 프롤레타리아 독재라는 말이 마르크스의 저작에서 가장 압축된 형태로 나타나는 자리다.

## 15년의 침묵, 그리고 축약본

마르크스는 원고를 발표하지 않았다. 비스마르크가 아이제나흐파 지도자들을 감옥에 넣던 시점이었고, 통합 자체를 깨뜨릴 뜻도 없었다.

1890년 할레 당대회가 고타강령 개정을 의제에 올리자 엥겔스는 더 이상 묻어 둘 수 없다고 판단했다. 1891년 1월 6일자 서문에서 그는 이 문서를 더 감춘다면 자신이 은폐의 죄를 짓는 것이라고 썼다. 당 지도부는 반대했고 엥겔스는 카우츠키의 『신시대』에 발표를 강행했다. 다만 실린 것은 축약본이다. 엥겔스 자신이 서문에 밝힌 대로 "몇몇 날카로운 인신 표현과 판단"은 점으로 대체되었고, 출판법 때문에 점으로만 남은 문장도 있으며, 완화한 표현은 대괄호에 넣었다. 인쇄된 텍스트는 원고 그 자체가 아니다.

## 이 문서가 남긴 것

레닌은 1917년 『국가와 혁명』 5장을 통째로 이 문서의 주석에 바쳤다. 마르크스가 '낮은 단계'라 부른 것에 사회주의를, '높은 단계'에 공산주의를 붙인 것이 그 장이고, 20세기 내내 쓰인 사회주의/공산주의 구분이 여기서 나왔다. 1936년 소련 헌법 12조가 "각자는 능력에 따라, 각자에게는 노동에 따라"를 사회주의의 원칙으로 명시한 것도 같은 계보다. 필요가 아니라 노동에 따라 — 낮은 단계의 공식을 헌법에 박아 넣은 것이다.

오늘 이 문서를 읽는 이유는 조금 다른 데 있다. 국가가 나서서 소유하고 보조하면 그것이 사회주의라는 생각은 1875년에 이미 노동자 정당의 강령이 되어 있었고, 마르크스는 그 자리에서 그것을 거부했다. 국가사회주의라는 말이 마르크스주의 전통에서 한 번도 자기 이름이 되지 못한 이유가 여기에 있다.""",
            "en": """## Written as a Warning, Not a Toast

In May 1875 two German workers' parties were about to merge at Gotha. On one side stood the Eisenachers — the Social Democratic Workers' Party of Wilhelm Liebknecht and August Bebel, close to Marx and Engels; on the other, the General German Workers' Association founded by Ferdinand Lassalle in 1863. Gathering the scattered forces under Bismarck's empire might have been welcome news. What came from London was not congratulation.

Marx wrote notes in the margin of the draft programme. The original title is 'Marginal Notes on the Programme of the German Workers' Party'; 'Critique of the Gotha Programme' came later. The manuscript, written in late April or early May 1875, went to Wilhelm Bracke to be passed round Geib, Auer, Bebel and Liebknecht and then returned to Marx. The congress effectively ignored it. The programme was adopted with minor changes and the new party became the Socialist Workers' Party of Germany.

## Four Targets

The critique proceeds by taking the programme's sentences in order. Its targets are four pieces of Lassalle's legacy.

**The undiminished proceeds of labour.** The first clause declares that 'the proceeds of labor belong undiminished with equal right to all members of society'. Marx cuts into the phrase before it: 'Labor is not the source of all wealth. Nature is just as much the source of use values… as labor.' He then lists what has to come off the total product first — replacement of means of production used up, a portion for expansion, reserve and insurance funds against accidents — and, from what remains for consumption, general administrative costs, common provision such as schools and health services, and funds for those unable to work. The 'undiminished' proceeds turn, unnoticed, into diminished ones.

**The iron law of wages.** The programme promised to abolish the wage system 'together with the iron law of wages'. Marx: nothing in that law is Lassalle's except the word 'iron', borrowed from Goethe.

**State-aided co-operatives.** The programme demanded producers' co-operatives founded with state aid, out of which the socialist organisation of labour would 'arise'. The answer is one sentence: 'It is worthy of Lassalle's imagination that with state loans one can build a new society just as well as a new railway!' Co-operatives count only 'insofar as they are the independent creations of the workers and not protégés either of the governments or of the bourgeois'.

**The free state.** The programme set out for a 'free state'. Marx asks what that is. Setting the state free is no aim of the workers: 'Freedom consists in converting the state from an organ superimposed upon society into one completely subordinate to it.'

## Two Phases, and One Sentence

Pushed to its end, the argument about distribution reaches the passage that divides communist society into two phases. In the lower phase, just out of capitalism and 'still stamped with the birthmarks of the old society', the producer draws back exactly the quantum of labour he gave, so bourgeois right survives. Only in the higher phase, after the enslaving subordination to the division of labour and the antithesis of mental and physical labour have vanished, can society inscribe on its banners: 'From each according to his ability, to each according to his needs!'

Part IV takes the state head-on: 'Between capitalist and communist society there lies the period of the revolutionary transformation of the one into the other. Corresponding to this is also a political transition period in which the state can be nothing but the revolutionary dictatorship of the proletariat.' It is the most compressed statement of the dictatorship of the proletariat anywhere in Marx.

## Fifteen Years of Silence, and an Abridged Text

Marx never published the manuscript. Bismarck was jailing Eisenacher leaders at the time, and he had no intention of wrecking the merger.

When the Halle congress of 1890 put revision of the Gotha Programme on the agenda, Engels judged that it could be withheld no longer. In his foreword of 6 January 1891 he wrote that to keep it back any longer would make him guilty of suppression. The leadership objected; Engels published it anyway, in Kautsky's Die Neue Zeit. What appeared, though, was an abridgement. As his foreword states, 'a few sharp personal expressions and judgments' were replaced by dots, some sentences survive only as dots because of the Press Law, and softened wordings were put in square brackets. The printed text is not the manuscript.

## What the Document Left Behind

In 1917 Lenin gave the whole of chapter 5 of The State and Revolution over to commentary on it. That chapter is where the 'lower phase' acquired the name socialism and the 'higher phase' the name communism, and the socialism/communism distinction used across the twentieth century comes from it. Article 12 of the 1936 Soviet Constitution, setting down 'From each according to his ability, to each according to his work' as the principle of socialism, belongs to the same line — according to work, not to needs: the formula of the lower phase, written into a constitution.

The reason to read the document now is slightly different. The idea that state ownership and state subsidy amount to socialism was already the programme of a workers' party in 1875, and Marx refused it on the spot. That is why 'state socialism' never became a name the Marxist tradition used for itself.""",
        },
    },
}

# ── 2. 공산주의의 낮은 단계와 높은 단계 ────────────────────────────────

PHASES = {
    "id": "lower-and-higher-phase-of-communism",
    "sources": [SRC_GOTHA_1, SRC_LENIN_5, SRC_CONSTITUTION_1936, SRC_GOTHA_INDEX],
    "patch": {
        "term": {
            "ko": "공산주의의 낮은 단계와 높은 단계",
            "en": "Lower and Higher Phase of Communism",
        },
        "original": "niedere und höhere Phase der kommunistischen Gesellschaft",
        "period": {"ko": "1875년~", "en": "1875–present"},
        "startYear": 1875,
        "category": "theory",
        "parentId": GOTHA["id"],
        "aliases": {
            "ko": [
                "공산주의의 낮은 단계와 높은 단계", "공산주의의 낮은 단계",
                "공산주의의 높은 단계", "낮은 단계와 높은 단계",
                "각자는 능력에 따라, 각자에게는 필요에 따라",
                "능력에 따라 일하고 필요에 따라 분배",
            ],
            "en": [
                "lower and higher phase of communism", "first phase of communist society",
                "higher phase of communist society",
                "from each according to his ability, to each according to his needs",
            ],
        },
        "people": ["karl-marx", "lenin", "stalin"],
        "definition": {
            "ko": (
                "『고타강령 비판』이 공산주의 사회를 두 단계로 갈라 본 구분. 자본주의에서 갓 나온 낮은 "
                "단계에서는 각자가 사회에 준 노동량만큼을 돌려받으므로 상품교환과 같은 등가 원리, 곧 "
                "'부르주아적 권리'가 그대로 남는다. 분업에 대한 예속과 정신·육체노동의 대립이 사라지고 "
                "노동이 삶의 첫째 욕구가 된 높은 단계에 가서야 '각자는 능력에 따라, 각자에게는 필요에 "
                "따라'가 가능해진다. 레닌이 『국가와 혁명』에서 낮은 단계를 사회주의로, 높은 단계를 "
                "공산주의로 이름 붙이면서 20세기의 표준 어법이 되었다."
            ),
            "en": (
                "The division of communist society into two phases made in the Critique of the "
                "Gotha Programme. In the lower phase, just emerged from capitalism, each producer "
                "draws back the quantity of labour he has given, so the equal-exchange principle "
                "of commodity society — 'bourgeois right' — survives intact. Only in the higher "
                "phase, once the enslaving subordination to the division of labour and the "
                "antithesis between mental and physical labour have gone and labour has become "
                "life's prime want, does 'from each according to his ability, to each according to "
                "his needs' become possible. Lenin's State and Revolution named the lower phase "
                "socialism and the higher phase communism, and that became the standard usage of "
                "the twentieth century."
            ),
        },
        "body": {
            "ko": """## 분배 논쟁의 막다른 곳에서 나온 구분

이 구분은 미래 사회의 설계도로 제시된 것이 아니다. 고타강령이 "공정한 분배"와 "감소되지 않은 노동수익"을 요구한 데 대한 반박을 밀고 나가다가 나온 결과물이다. 무엇이 공정한 분배인가를 따지려면 어떤 사회의 분배인지를 먼저 말해야 하고, 그러자면 자본주의에서 막 나온 사회와 자기 발로 선 사회를 갈라야 했다.

## 낮은 단계 — 노동증서와 부르주아적 권리

낮은 단계의 사회는 "경제적·도덕적·정신적으로 낡은 사회의 모반(母斑)을 온몸에 지닌" 채 나온다. 여기서 생산자는 공제분을 뺀 뒤 자신이 제공한 노동량을 증명하는 증서를 받고, 그 증서로 사회의 소비 재고에서 같은 양의 노동이 든 물건을 꺼내 간다. 개인 사이의 교환은 사라졌지만 원리는 남는다. 같은 양의 노동을 다른 형태로 되돌려 받는다는 점에서 상품교환을 규제하던 등가 원리가 그대로 작동한다.

마르크스는 이것을 "부르주아적 권리"라 부른다. 계급 차이는 인정하지 않지만 개인의 타고난 능력 차이는 자연적 특권으로 묵인하며, 결혼했는지 아이가 몇인지는 아예 셈에 넣지 않는다. "이 평등한 권리는 불평등한 노동에 대한 불평등한 권리다." 그렇다고 결함이라 부르며 고칠 수 있는 것도 아니다. "권리는 결코 사회의 경제적 구조와 그것이 제약하는 문화 발전보다 높을 수 없다."

## 높은 단계 — 좁은 지평 너머

높은 단계의 조건은 분배 방식이 아니라 노동 자체의 변화로 서술된다. 개인이 분업에 노예처럼 매이는 상태가 끝나고, 정신노동과 육체노동의 대립이 사라지고, 노동이 생계 수단을 넘어 삶의 첫째 욕구가 되고, 개인의 전면적 발전과 함께 생산력이 자라 협동적 부의 샘이 더 넘치게 흐를 때. 그때에야 부르주아적 권리의 좁은 지평을 온전히 넘어설 수 있다.

## 레닌의 이름 붙이기

『국가와 혁명』 5장에서 레닌은 "흔히 사회주의라 불리는 것을 마르크스는 공산주의 사회의 '첫 번째' 또는 낮은 단계라 불렀다"고 정리한다. 사회주의와 공산주의를 별개의 두 이념이 아니라 한 사회의 두 성숙도로 놓은 것이다. 같은 장에서 그는 낮은 단계에 대해 한 걸음 더 나간 표현을 쓴다. 분배를 규율할 법이 부르주아적 권리뿐인 한 그것을 집행할 장치도 남아야 하므로, "공산주의 아래에서 한동안 부르주아적 권리뿐 아니라 부르주아지 없는 부르주아 국가까지 남는다."

## 헌법에 박힌 낮은 단계

1936년 소련 헌법 12조는 사회주의의 원칙을 "각자는 능력에 따라, 각자에게는 노동에 따라"로 명시했다. 필요가 아니라 노동에 따라 — 마르크스가 낮은 단계의 한계라고 지적한 바로 그 공식이 국가의 최고 규범이 된 것이다. 이 조항은 그 자체로 모순이 아니다. 낮은 단계임을 인정한 서술이기 때문이다. 다만 이후 소련에서 그 인정은 자주 목표의 연기로 읽혔고, 부르주아적 권리의 존속은 설명되기보다 관리되었다.

## 읽을 때의 주의

두 단계는 시간표가 아니다. 마르크스는 연도를 말하지 않았고 레닌도 "높은 단계가 언제 오는지 우리는 알지 못하며 알 수도 없다"고 못박았다. 이 구분이 실제로 하는 일은 예언이 아니라 진단이다. 어떤 사회가 아직 노동에 따라 나누고 있다면 그 사회는 아직 낮은 단계이며, 거기 남은 불평등은 도덕의 실패가 아니라 그 단계의 경제적 조건이라는 것.""",
            "en": """## A Distinction Reached at the End of an Argument About Distribution

The two phases are not offered as a blueprint. They come out of pressing the reply to the Gotha Programme's demand for 'fair distribution' and 'undiminished proceeds of labour'. To ask what distribution is fair, you first have to say in what society — which forced a split between a society just out of capitalism and one standing on its own foundations.

## The Lower Phase — Labour Certificates and Bourgeois Right

The lower phase emerges 'in every respect, economically, morally, and intellectually, still stamped with the birthmarks of the old society'. The producer receives, after the deductions, a certificate for the labour he has furnished, and draws from the social stock means of consumption costing the same amount of labour. Exchange between individuals is gone; the principle is not. The same quantity of labour returns in another form, so the rule that governed commodity exchange still governs.

Marx calls this 'bourgeois right'. It recognises no class differences, yet tacitly accepts unequal individual endowment as a natural privilege, and takes no account of whether one worker is married or has more children than another. 'This equal right is an unequal right for unequal labour.' Nor is it a fault that can simply be corrected: 'Right can never be higher than the economic structure of society and its cultural development conditioned thereby.'

## The Higher Phase — Past the Narrow Horizon

The condition for the higher phase is stated as a change in labour itself, not in the method of distribution: after the enslaving subordination of the individual to the division of labour has vanished, after the antithesis between mental and physical labour has gone, after labour has become not merely a means of life but life's prime want, and after the productive forces have grown with the all-round development of the individual so that the springs of co-operative wealth flow more abundantly. Only then can the narrow horizon of bourgeois right be crossed in its entirety.

## Lenin's Naming

In chapter 5 of The State and Revolution Lenin sets it down: 'What is usually called socialism was termed by Marx the "first", or lower, phase of communist society.' Socialism and communism become two degrees of maturity of one society rather than two doctrines. In the same chapter he pushes the lower phase one step further: so long as the only rules for distribution are those of bourgeois right, something must enforce them, and therefore 'under communism there remains for a time not only bourgeois law, but even the bourgeois state, without the bourgeoisie!'

## The Lower Phase Written Into a Constitution

Article 12 of the 1936 Soviet Constitution set down the principle of socialism as 'From each according to his ability, to each according to his work'. According to work, not to needs — precisely the formula Marx had identified as the limit of the lower phase, now the supreme norm of a state. The article is not in itself a contradiction; it is an admission of which phase the country was in. In practice, though, that admission was often read in the USSR as a deferral of the goal, and the survival of bourgeois right was managed more than it was explained.

## How to Read It

The two phases are not a timetable. Marx named no dates, and Lenin insisted that when the higher phase arrives 'we do not and cannot know'. What the distinction actually does is diagnostic rather than prophetic: a society still distributing according to work is still in the lower phase, and the inequality left in it is a condition of that phase's economy rather than a failure of morals.""",
        },
    },
}

# ── 3. 노동전수익권 ────────────────────────────────────────────────────

PROCEEDS = {
    "id": "undiminished-proceeds-of-labour",
    "sources": [SRC_GOTHA_1, SRC_MENGER, SRC_LASSALLE, SRC_LENIN_5],
    "patch": {
        "term": {"ko": "노동전수익권", "en": "Undiminished Proceeds of Labour"},
        "original": "der unverkürzte Arbeitsertrag",
        "period": {"ko": "1860년대–19세기 말", "en": "1860s–late 19th century"},
        "startYear": 1860,
        "endYear": 1900,
        "category": "theory",
        "parentId": GOTHA["id"],
        "aliases": {
            "ko": ["노동전수익권", "노동수익 전액", "감소되지 않은 노동수익"],
            "en": [
                "undiminished proceeds of labour", "undiminished proceeds of labor",
                "right to the whole produce of labour", "unverkürzter Arbeitsertrag",
            ],
        },
        "people": ["karl-marx", "ferdinand-lassalle"],
        "definition": {
            "ko": (
                "노동자가 자기 노동이 낳은 수익 전부를 돌려받아야 한다는 라살레파의 요구. 고타강령 첫 "
                "조항에 '노동의 수익은 감소되지 않은 채 평등한 권리로 사회의 모든 구성원에게 "
                "귀속된다'는 형태로 들어갔다. 마르크스는 이 요구가 두 번 무너진다고 보았다. 총생산물에서 "
                "생산수단 대체분·확대분·예비기금을 먼저 떼어야 하고, 남은 소비 몫에서도 관리비, 학교와 "
                "보건, 노동 능력을 잃은 이들을 위한 기금을 떼어야 한다. 나아가 분배는 생산조건의 "
                "분배에서 나오는 결과이므로, 분배를 앞세우는 사회주의는 문제를 거꾸로 든 것이라고 했다."
            ),
            "en": (
                "The Lassallean demand that the worker receive the entire proceeds of his labour, "
                "which entered the Gotha Programme as the claim that 'the proceeds of labor belong "
                "undiminished with equal right to all members of society'. Marx showed the demand "
                "collapsing twice over: from the total product must first come replacement of "
                "means of production, a portion for expansion and reserve funds against accidents; "
                "and from what remains for consumption come administrative costs, common provision "
                "such as schools and health services, and funds for those unable to work. Beyond "
                "the arithmetic he argued that distribution follows from the distribution of the "
                "conditions of production, so a socialism built on distribution has the question "
                "upside down."
            ),
        },
        "body": {
            "ko": """## 어디서 온 요구인가

노동자가 만든 것은 노동자의 것이라는 주장은 19세기 사회주의의 공용 자산이었다. 라살레가 그것을 강령 언어로 굳혔고, 안톤 멩거가 1886년 『역사적 서술로 본 노동전수익권』을 써서 이 요구의 계보를 정리했다. 한국어의 '노동전수익권'이라는 딱딱한 낱말은 이 계보를 옮기는 과정에서 자리 잡았다.

요구 자체는 직관적이다. 자본가가 가져가는 몫이 착취라면, 착취를 없앤 사회에서 노동자는 자기 노동의 결과를 통째로 받아야 하지 않겠는가.

## 마르크스의 공제 목록

『고타강령 비판』 1부는 이 직관을 계산으로 반박한다. 협동조합적 사회의 총노동수익은 총사회적 생산물이다. 여기서 먼저 떼어야 할 것이 있다.

1. 소모된 생산수단을 대체할 몫
2. 생산 확대를 위한 추가분
3. 사고와 자연재해에 대비한 예비기금 또는 보험기금

남은 것이 소비수단이다. 그런데 이것을 개인들에게 나누기 전에 또 떼어야 한다.

4. 생산에 속하지 않는 일반 관리비 — 새 사회가 발전할수록 줄어든다
5. 학교와 보건처럼 공동으로 충족되는 수요 — 새 사회가 발전할수록 늘어난다
6. 노동 능력을 잃은 사람들을 위한 기금

이 목록을 지나면 "감소되지 않은" 수익이라는 말은 이미 무너져 있다. 마르크스의 표현으로 그것은 "어느새 '감소된' 수익으로 바뀌어" 있고, 다만 개인으로서 빼앗긴 것이 사회 구성원으로서 직접·간접으로 그에게 돌아올 뿐이다. 여기서 한 걸음 더 나가면 '노동수익'이라는 말 자체가 뜻을 잃는다. 공동소유 위에 선 사회에서 생산자들은 생산물을 교환하지 않고, 개인의 노동은 총노동의 구성 부분으로 직접 존재하기 때문이다.

## 진짜 반론은 산수가 아니다

공제 목록은 예비 작업이다. 이 대목의 결론은 따로 있다.

"소위 분배를 두고 야단법석을 떨며 거기에 주된 강조를 두는 것은 애초에 잘못이었다."

소비수단의 분배는 생산조건 자체의 분배에서 나오는 결과다. 물질적 생산조건이 자본과 토지의 형태로 비노동자의 손에 있고 대중은 노동력이라는 개인적 조건만 가진 사회에서는, 오늘날 같은 분배가 자동으로 따라 나온다. 생산의 물질적 조건이 노동자 자신의 협동조합적 소유가 되면 다른 분배가 마찬가지로 자동으로 따라 나온다. 마르크스가 "속류 사회주의"라 부른 것은 이 순서를 뒤집어 분배를 생산양식과 무관한 독립 변수로 다루는 사고방식이다.

## 그 요구가 남긴 자리

노동전수익권은 강령에서 사라졌지만 사고 습관으로는 살아남았다. 소유 구조를 건드리지 않은 채 몫을 다시 나누는 것으로 문제를 풀려는 모든 제안이 이 계보에 놓인다. 『고타강령 비판』의 대답은 그 제안들이 틀렸다는 것이 아니라, 그것으로 사회주의를 정의할 수는 없다는 것이다. 무엇을 얼마나 나누는가보다 생산조건이 누구의 것인가를 먼저 묻지 않으면, 분배 공식은 언제든 국가가 관리하는 시혜로 미끄러진다.""",
            "en": """## Where the Demand Came From

That what the worker makes belongs to the worker was common property of nineteenth-century socialism. Lassalle hardened it into programmatic language, and in 1886 Anton Menger traced the whole claim in The Right to the Whole Produce of Labour. The stiff Korean coinage 노동전수익권 dates from the translation of that lineage.

The demand is intuitive enough. If the capitalist's share is exploitation, then in a society that has abolished exploitation the worker should surely receive the product of his labour entire.

## Marx's List of Deductions

Part I of the Critique answers the intuition with arithmetic. The co-operative proceeds of labour are the total social product, and from it must first be deducted:

1. cover for replacement of the means of production used up
2. an additional portion for the expansion of production
3. reserve or insurance funds against accidents and natural calamities

What remains is means of consumption — but before that is divided among individuals, further deductions are made:

4. general costs of administration not belonging to production, which shrink as the new society develops
5. what is meant for the common satisfaction of needs, such as schools and health services, which grow as it develops
6. funds for those unable to work

By the end of the list the word 'undiminished' has already collapsed. In Marx's phrase the proceeds 'have already unnoticeably become converted into the "diminished" proceeds', with the qualification that what the producer loses as a private individual comes back to him directly or indirectly as a member of society. One step further and the phrase 'proceeds of labour' loses its meaning altogether, since in a society resting on common ownership the producers do not exchange their products and individual labour exists directly as a component of total labour.

## The Real Objection Is Not Arithmetic

The deductions are preliminary. The conclusion of the passage lies elsewhere:

'It was in general a mistake to make a fuss about so-called distribution and put the principal stress on it.'

Any distribution of means of consumption is a consequence of the distribution of the conditions of production themselves. Where the material conditions of production are in the hands of non-workers as capital and land while the masses own only labour power, present-day distribution follows automatically; where the material conditions are the co-operative property of the workers, a different distribution follows just as automatically. What Marx called 'vulgar socialism' is the habit of reversing this order and treating distribution as independent of the mode of production.

## What the Demand Left Behind

The undiminished proceeds vanished from the programmes but survived as a habit of thought. Every proposal that leaves ownership untouched and solves the problem by redividing the shares stands in this line. The Critique's answer is not that such proposals are wrong, but that socialism cannot be defined by them: unless you ask whose the conditions of production are before you ask how much is shared out, a distributive formula slides easily into a benefit administered by the state.""",
        },
    },
}

# ── 4. 임금철칙 ──────────────────────────────────────────────────────

IRON_LAW = {
    "id": "iron-law-of-wages",
    "sources": [
        SRC_GOTHA_2, SRC_IRON_LAW, SRC_LASSALLE,
        SRC_KO_IRON_LAW_NAME, SRC_KO_IRON_LAW_DICT, SRC_KO_IRON_LAW_GOTHA,
    ],
    "patch": {
        "term": {"ko": "임금철칙", "en": "Iron Law of Wages"},
        "original": "ehernes Lohngesetz",
        "period": {"ko": "1863–1891", "en": "1863–1891"},
        "startYear": 1863,
        "endYear": 1891,
        "category": "theory",
        "parentId": GOTHA["id"],
        "aliases": {
            "ko": ["임금철칙", "임금의 철칙", "임금철칙설", "철의 임금법칙"],
            "en": ["iron law of wages", "ehernes Lohngesetz"],
        },
        "people": ["ferdinand-lassalle", "karl-marx"],
        "definition": {
            "ko": (
                "임금은 노동자의 생존과 번식에 필요한 최소한으로 언제나 되돌아간다는 라살레의 주장. "
                "튀르고와 리카도의 생존임금론, 맬서스의 인구론에서 끌어온 것이고 '철칙'이라는 낱말은 "
                "괴테의 '위대하고 영원한 철의 법칙'에서 빌려 왔다. 임금 인상 투쟁이 무의미하므로 국가 "
                "보조 생산협동조합으로 임금노동 자체를 벗어나야 한다는 그의 결론을 떠받치는 기둥이었다. "
                "마르크스는 이 법칙에서 라살레의 것은 낱말 하나뿐이며, 근거가 맬서스라면 임금노동을 백 "
                "번 폐지해도 법칙은 살아남는다고 반박했다. 임금은 노동의 값이 아니라 노동력의 값이라는 "
                "것이 그의 정정이다."
            ),
            "en": (
                "Lassalle's claim that wages always return to the minimum required for the worker's "
                "subsistence and reproduction. It descends from the subsistence-wage doctrine of "
                "Turgot and Ricardo and from Malthus's population theory, and the word 'iron' was "
                "taken from Goethe's 'great, eternal iron laws'. It was the pillar holding up his "
                "conclusion that struggle over wages is pointless and that workers must escape wage "
                "labour itself through state-financed producers' co-operatives. Marx replied that "
                "nothing in the law is Lassalle's except the word, and that if its Malthusian "
                "grounding held, the law would survive the abolition of wage labour a hundred times "
                "over. His correction: wages are not the price of labour but the price of labour "
                "power."
            ),
        },
        "body": {
            "ko": """## 결론을 떠받치기 위해 필요했던 법칙

라살레의 정치 전략은 단순하다. 노동조합의 임금 투쟁으로는 아무것도 얻지 못하니, 보통선거권으로 국가를 움직여 국가 보조 생산협동조합을 세우고 임금노동 자체에서 걸어 나가야 한다. 이 결론이 성립하려면 임금 투쟁이 원리상 무의미하다는 것이 먼저 증명되어야 한다. 그 증명을 맡은 것이 라살레 자신의 표현으로 '철과 같은, 잔혹한 법칙(das eiserne und grausame Gesetz)'이었다.

내용은 라살레의 발명이 아니다. 임금이 생존 수준으로 수렴한다는 생각은 튀르고와 리카도에게 있었고, 그 배후의 인구 압력 논리는 맬서스의 것이다. 라살레가 더한 것은 이름이다. '철칙(鐵則)'의 쇠는 괴테의 시 「신적인 것」에 나오는 "위대하고 영원한 철의 법칙(eherne Gesetze)"에서 왔다.

## 마르크스의 세 갈래 반박

『고타강령 비판』 2부는 강령이 임금제도를 "임금철칙과 함께" 폐지하겠다고 쓴 대목을 겨눈다.

**첫째, 낱말뿐이다.** "이 '임금철칙'에서 라살레의 것은 괴테의 '위대하고 영원한 철의 법칙'에서 빌려 온 '철칙'이라는 낱말뿐이라는 사실은 잘 알려져 있다." 그 낱말은 신도들이 서로를 알아보는 표지일 뿐이라고 마르크스는 덧붙인다.

**둘째, 근거를 받아들이면 결론이 뒤집힌다.** 라살레의 뜻대로 이 법칙을 취하면 그 근거인 맬서스 인구론도 함께 취해야 한다. 그런데 그 이론이 옳다면 임금노동을 백 번 폐지해도 법칙은 남는다. 법칙이 지배하는 것이 임금노동 체제만이 아니라 모든 사회 체제가 되기 때문이다. 실제로 경제학자들은 50년 넘게 그 논리로 사회주의는 빈곤을 없애지 못하고 사회 전면에 고루 퍼뜨릴 뿐이라고 논증해 왔다.

**셋째, 임금이 무엇의 값인지가 틀렸다.** 라살레 사후 당 안에 자리 잡은 과학적 인식은 임금이 노동의 가치나 가격이 아니라 노동력의 가치·가격의 가면 쓴 형태라는 것이다. 이 정정과 함께 지금까지의 부르주아적 임금관과 그에 대한 비판이 통째로 폐기되었는데, 강령은 그 인식이 퍼진 뒤에 다시 라살레의 교리로 돌아갔다.

마르크스의 비유가 이 대목을 끝낸다. 노예제의 비밀을 마침내 알아채고 반란을 일으킨 노예들 가운데 낡은 관념에 매인 한 사람이 반란의 강령에 이렇게 써넣는 격이라는 것이다. "노예제는 폐지되어야 한다. 노예제 아래서 노예에게 주는 먹이가 일정한 낮은 최대치를 넘을 수 없기 때문이다."

## 강령에서 사라지다

라살레의 법칙은 1891년 에르푸르트 강령에서 자취를 감춘다. 1875년 고타강령에 남아 있던 라살레주의 공식들을 카우츠키와 베른슈타인이 새로 쓴 강령이 걷어낸 것이다. 『고타강령 비판』이 발표된 것도 바로 그 개정 논의가 진행되던 1891년이었다.

## 한국어 이름

원어는 ehernes Lohngesetz다. 한국어에서 자리 잡은 이름은 임금철칙(賃金鐵則)이고, 임금의 철칙과 임금철칙설도 함께 쓰인다. 영어 iron law of wages를 글자대로 옮긴 '철의 임금법칙'은 통용되는 형태가 아니다. 이 항목은 임금철칙을 표제어로 삼고 나머지를 별칭으로 등록해, 어느 표기로 쓴 글에서든 이 항목으로 이어지게 했다.

## 왜 여전히 다뤄야 하는가

임금이 생존선으로 눌린다는 서술 자체는 사라지지 않았다. 사라져야 하는 것은 그것을 자연법칙으로 다루는 방식이다. 임금을 노동력의 값으로 보면 그 값은 역사적·도덕적 요소를 포함하고 계급 간 힘 관계에 따라 움직인다. 법칙이 아니라 관계이므로 투쟁의 대상이 된다. 라살레의 법칙이 임금 투쟁을 무의미하게 만들고 그 자리에 국가를 불러들였다면, 마르크스의 정정은 임금 투쟁을 되살리면서 동시에 그것만으로는 임금노동 체제를 넘지 못한다고 말한다.""",
            "en": """## A Law Needed to Hold Up a Conclusion

Lassalle's political strategy was simple: trade-union struggle over wages wins nothing, so the workers must use universal suffrage to move the state to finance producers' co-operatives and walk out of wage labour altogether. For that conclusion to stand, wage struggle has to be pointless in principle — and the proof was assigned to the 'iron and cruel law'.

The content was not his invention. That wages converge on subsistence is in Turgot and Ricardo, and the population pressure behind it is Malthus's. What Lassalle added was the name: 'iron' comes from Goethe's poem Das Göttliche and its 'great, eternal iron laws'.

## Marx's Three Lines of Attack

Part II of the Critique goes after the clause promising to abolish the wage system 'together with the iron law of wages'.

**First, only the word is his.** 'It is well known that nothing of the "iron law of wages" is Lassalle's except the word "iron" borrowed from Goethe's "great, eternal iron laws".' The word, Marx adds, is a label by which the true believers recognise one another.

**Second, accept the grounding and the conclusion inverts.** Take the law with Lassalle's stamp on it and you must take his substantiation too — the Malthusian theory of population. But if that theory is correct, the law survives the abolition of wage labour a hundred times over, because it then governs not the wage system alone but every social system. Economists had been arguing on exactly that basis for fifty years that socialism cannot abolish poverty but only spread it evenly over the whole surface of society.

**Third, it misidentifies what wages are the price of.** The understanding established in the party after Lassalle's death was that wages are not the value or price of labour but a masked form of the value or price of labour power. With that correction the whole bourgeois conception of wages, and all the criticism directed at it, went overboard — and the programme, written afterwards, returned to Lassalle's dogma.

Marx closes with an image: it is as if, among slaves who have at last got behind the secret of slavery and broken into rebellion, one still in thrall to obsolete notions inscribed on the programme of the rebellion, 'Slavery must be abolished because the feeding of slaves in the system of slavery cannot exceed a certain low maximum!'

## Gone From the Programme

Lassalle's law disappears from the Erfurt Programme of 1891, which Kautsky and Bernstein wrote to clear out the Lassallean formulas left standing at Gotha in 1875. The Critique itself was published in that same year, in the middle of the revision debate.

## The Korean Name

The German is ehernes Lohngesetz. Korean settled on 임금철칙 (賃金鐵則), with 임금의 철칙 and 임금철칙설 also in use; 철의 임금법칙, a word-for-word rendering of the English name, is not current. This entry takes 임금철칙 as its headword and registers the others as aliases, so prose written any of those ways still links here.

## Why It Still Has to Be Handled

The observation that wages are pressed toward subsistence has not gone away. What has to go is treating it as a law of nature. Read wages as the price of labour power and that price contains a historical and moral element and moves with the balance of class forces — a relation, not a law, and therefore something to fight over. Where Lassalle's law made wage struggle meaningless and called in the state to fill the gap, Marx's correction restores wage struggle while insisting that it does not by itself carry anyone past the wage system.""",
        },
    },
}

# ── 5. 공상에서 과학으로 ──────────────────────────────────────────────

UTOPIAN_SCIENTIFIC = {
    "id": "socialism-utopian-and-scientific",
    "sources": [
        SRC_SUS_INDEX, SRC_SUS_1, SRC_SUS_2, SRC_SUS_3, SRC_ANTI_DUHRING,
        SRC_LENIN_1, SRC_W_LIEBKNECHT,
    ],
    "patch": {
        "term": {"ko": "공상에서 과학으로", "en": "Socialism: Utopian and Scientific"},
        "original": "Die Entwicklung des Sozialismus von der Utopie zur Wissenschaft",
        "period": {"ko": "1880년", "en": "1880"},
        "startYear": 1880,
        "endYear": 1880,
        "category": "theory",
        "aliases": {
            "ko": [
                "공상에서 과학으로", "공상적 사회주의와 과학적 사회주의",
                "사회주의의 공상에서 과학으로의 발전",
            ],
            "en": [
                "Socialism: Utopian and Scientific", "Socialisme utopique et Socialisme scientifique",
                "Die Entwicklung des Sozialismus von der Utopie zur Wissenschaft",
            ],
        },
        "people": ["friedrich-engels", "karl-marx", "lenin", "wilhelm-liebknecht"],
        "definition": {
            "ko": (
                "엥겔스가 『반뒤링론』(1878)에서 세 개 장을 뽑아 엮은 소책자. 라파르그의 요청으로 "
                "프랑스어로 옮겨져 1880년 『사회주의 평론』 3·4·5월호에 실렸고, 1883년 독일어 원어판, "
                "1892년 영어판까지 열 개 언어로 퍼졌다. 엥겔스 자신이 『공산당 선언』이나 『자본론』보다 "
                "더 자주 번역된 책이라고 적었다. 1부는 생시몽·푸리에·오언을, 2부는 변증법과 유물사관을, "
                "3부는 자본주의의 모순과 국가를 다룬다. 비스마르크식 국유화를 '사이비 사회주의'라 부른 "
                "각주와 '국가는 폐지되는 것이 아니라 사멸한다'는 문장이 여기에 있다."
            ),
            "en": (
                "The pamphlet Engels drew out of three chapters of Anti-Dühring (1878). At Paul "
                "Lafargue's request it was translated into French and printed in the March, April "
                "and May 1880 issues of Revue Socialiste; the German original followed in 1883 and "
                "the English edition in 1892, by which point it circulated in ten languages. Engels "
                "himself noted that no other socialist work, not even the Communist Manifesto or "
                "Capital, had been translated so often. Part I treats Saint-Simon, Fourier and "
                "Owen; Part II dialectics and the materialist conception of history; Part III the "
                "contradictions of capitalism and the state. It carries the footnote calling "
                "Bismarckian nationalisation 'a kind of spurious Socialism' and the sentence that "
                "the state 'is not abolished. It dies out.'"
            ),
        },
        "body": {
            "ko": """## 반뒤링론에서 떨어져 나온 소책자

1877년부터 엥겔스는 베를린의 오이겐 뒤링을 반박하는 연재를 썼다. 뒤링의 체계가 독일 사민당 안에서 마르크스주의를 밀어내고 있었기 때문이다. 그 연재를 묶은 것이 『오이겐 뒤링 씨의 과학 변혁』, 흔히 『반뒤링론』(1878)이다.

폴 라파르그가 그중 몇 장을 따로 내자고 청했다. 엥겔스가 세 개 장을 골라 소책자로 엮고 라파르그가 옮겨 1880년 『사회주의 평론』 3·4·5월호에 「공상적 사회주의와 과학적 사회주의」로 실었다. 폴란드어·에스파냐어판이 이 프랑스어본에서 나왔고, 1883년 독일 동지들이 원어판을 냈으며, 이탈리아어·러시아어·덴마크어·네덜란드어·루마니아어판이 독일어본에서 나왔다. 1892년 영어판 서문에서 엥겔스는 이 작은 책이 열 개 언어로 돌고 있다며 이렇게 적었다. "다른 어떤 사회주의 저작도, 1848년의 우리 『공산당 선언』이나 마르크스의 『자본론』조차도 이만큼 자주 번역되지는 않았다."

말하자면 이 소책자가 마르크스주의가 대중에게 도달한 실제 통로다. 20세기 사회주의 운동의 상당수는 『자본론』이 아니라 이 100쪽 남짓한 책으로 입문했다.

## 1부 — 세 사람의 공상

첫 장은 생시몽·푸리에·오언을 다룬다. 흔한 오해와 달리 조롱의 장이 아니다. 엥겔스는 세 사람이 프랑스 혁명이 약속한 이성의 왕국이 실제로는 "쓰라리게 실망스러운 희화"로 나타난 자리에서 등장했다고 설명한다. 그들이 해법을 머릿속에서 끌어내 밖에서 사회에 부과하려 한 것은 개인의 결함이 아니라 자본주의와 계급 대립이 아직 미숙했던 조건의 반영이다. "우리는 그 환상적 외피를 뚫고 도처에서 터져 나오는 놀랍도록 위대한 사상과 그 맹아를 기뻐한다"는 것이 엥겔스의 태도다.

## 2부 — 왜 '과학적'인가

둘째 장은 형이상학적 사고와 변증법을 대비하고 헤겔에서 유물론으로 이어지는 길을 요약한 뒤, 사회주의가 과학이 된 지점을 두 발견으로 못박는다. 유물사관과 잉여가치를 통한 자본주의적 생산의 비밀 해명. "이 두 위대한 발견을 우리는 마르크스에게 빚지고 있다. 이 발견들과 함께 사회주의는 과학이 되었다."

## 3부 — 국유화는 해결이 아니다

셋째 장이 이 항목을 국가 문제와 잇는 부분이다. 엥겔스는 주식회사와 트러스트, 그리고 국가에 의한 생산수단 인수를 자본주의 자체가 자기 형태를 넘어서는 징후로 읽는다. 그러나 곧바로 못을 박는다.

"근대국가는 그 형태가 무엇이든 본질적으로 자본주의적 기계이며, 자본가들의 국가이자 총국민자본의 이상적 총괄자다. 생산력을 더 많이 인수할수록 국가는 더욱 실제의 국민자본가가 되고 더 많은 시민을 착취한다. 노동자는 여전히 임금노동자, 프롤레타리아로 남는다. 자본관계는 폐기되지 않는다. 오히려 극한까지 밀린다."

그 유명한 각주가 여기에 붙는다. 비스마르크가 국유화에 나선 뒤로 모든 국가 소유를 사회주의적이라 부르는 "일종의 사이비 사회주의"가 생겨났는데, 국가가 담배 전매를 하는 것이 사회주의라면 나폴레옹과 메테르니히도 사회주의의 창시자에 넣어야 한다. 벨기에가 정치적·재정적 이유로 철도를 놓고 비스마르크가 전시에 다루기 편하려고 프로이센 철도를 사들인 것이 사회주의라면, 왕립 도자기 공장과 군대의 연대 재봉사까지 사회주의 기관이 된다.

## 국가의 사멸

같은 장의 결론부에 국가에 관한 가장 많이 인용된 문장들이 있다. 프롤레타리아가 정치권력을 잡아 생산수단을 국가 소유로 전환하는 그 행위로 프롤레타리아는 스스로를 폐기하고 국가를 국가로서 폐기한다. 사회 전체의 이름으로 생산수단을 장악하는 첫 행위는 동시에 국가로서의 마지막 독자적 행위다. 그 뒤로 "사람에 대한 통치는 사물의 관리와 생산 과정의 지도로 대체된다." 그리고, "국가는 '폐지'되는 것이 아니다. 그것은 사멸한다."

레닌은 1917년 이 대목을 다시 읽으며, 사멸하는 것은 혁명 뒤에 남은 프롤레타리아 반(半)국가이고 부르주아 국가는 혁명으로 폐지되는 것이라고 정정했다. 그 정정은 이 소책자를 개량주의의 알리바이로 쓰던 독법을 겨눈 것이었다.

## 한국어에 남긴 이름

'공상적 사회주의'와 '과학적 사회주의'라는 짝은 이 소책자의 제목에서 왔다. 한국어 번역본은 「공상에서 과학으로」, 「공상적 사회주의와 과학적 사회주의」, 「사회주의의 공상에서 과학으로의 발전」 등으로 나왔는데, 원제 Die Entwicklung des Sozialismus von der Utopie zur Wissenschaft에 가장 가까운 것은 마지막이고, 가장 널리 불리는 것은 첫째다.""",
            "en": """## A Pamphlet Broken Off From Anti-Dühring

From 1877 Engels wrote a serial refuting Eugen Dühring of Berlin, whose system was displacing Marxism inside the German party. Collected, it became Herr Eugen Dühring's Revolution in Science — Anti-Dühring (1878).

Paul Lafargue asked for some of it separately. Engels arranged three chapters as a pamphlet, Lafargue translated it, and it appeared in the March, April and May 1880 issues of Revue Socialiste as Socialisme utopique et Socialisme scientifique. Polish and Spanish editions came from the French, German friends brought out the original in 1883, and Italian, Russian, Danish, Dutch and Romanian versions followed from the German. In the 1892 English preface Engels noted that the little book was circulating in ten languages: 'I am not aware that any other Socialist work, not even our Communist Manifesto of 1848, or Marx's Capital, has been so often translated.'

This pamphlet, in other words, was the actual channel by which Marxism reached a mass readership. Much of the twentieth-century socialist movement came in through these hundred-odd pages rather than through Capital.

## Part I — Three Men's Utopias

The first chapter treats Saint-Simon, Fourier and Owen, and against the common impression it is not a chapter of mockery. Engels places the three at the point where the kingdom of reason promised by the French Revolution had turned out to be a set of 'bitterly disappointing caricatures'. That they drew their solutions out of the human brain and sought to impose them on society from outside is a reflection of immature capitalist and class conditions, not a personal failing. His own attitude is explicit: 'we delight in the stupendously grand thoughts and germs of thought that everywhere break out through their phantastic covering.'

## Part II — What Makes It 'Scientific'

The second chapter sets metaphysical thinking against dialectics, summarises the road from Hegel to materialism, and fixes the point at which socialism became a science in two discoveries — the materialist conception of history, and the secret of capitalist production laid open through surplus value. 'These two great discoveries… we owe to Marx. With these discoveries, Socialism became a science.'

## Part III — Nationalisation Is Not the Solution

The third chapter is what joins this entry to the question of the state. Engels reads joint-stock companies, trusts and state takeovers as signs of capitalism outgrowing its own form — and then nails the point down:

'The modern state, no matter what its form, is essentially a capitalist machine — the state of the capitalists, the ideal personification of the total national capital. The more it proceeds to the taking over of productive forces, the more does it actually become the national capitalist, the more citizens does it exploit. The workers remain wage-workers — proletarians. The capitalist relation is not done away with. It is, rather, brought to a head.'

The famous footnote hangs here. Since Bismarck went in for state ownership, a 'kind of spurious Socialism' has arisen that declares all state ownership socialistic — but if the state taking over the tobacco trade is socialist, then Napoleon and Metternich must be numbered among the founders of socialism. If Belgium building its railways for political and financial reasons, and Bismarck buying the Prussian lines to have them in hand in case of war, were socialist measures, then the royal porcelain manufacture and the army's regimental tailor would be socialist institutions too.

## The State Dies Out

The most quoted sentences on the state close the same chapter. In seizing political power and turning the means of production into state property the proletariat abolishes itself as proletariat and abolishes the state as state. That first act — taking possession of the means of production in the name of society — is at the same time its last independent act as a state. Thereafter 'the government of persons is replaced by the administration of things, and by the conduct of processes of production'. And: 'The State is not "abolished". It dies out.'

Rereading the passage in 1917, Lenin corrected the usual gloss: what withers away is the proletarian semi-state left after the revolution, while the bourgeois state is abolished by the revolution itself. The correction was aimed at readings that used this pamphlet as an alibi for reformism.

## The Names It Left in Korean

The pair 공상적 사회주의 / 과학적 사회주의 comes from this pamphlet's title. Korean editions have appeared as 공상에서 과학으로, 공상적 사회주의와 과학적 사회주의, and 사회주의의 공상에서 과학으로의 발전; the last is closest to Die Entwicklung des Sozialismus von der Utopie zur Wissenschaft, and the first is what everyone says.""",
        },
    },
}

# ── 6. 공상적 사회주의 ────────────────────────────────────────────────

UTOPIAN = {
    "id": "utopian-socialism",
    "sources": [SRC_SUS_1, SRC_SUS_INDEX],
    "patch": {
        "term": {"ko": "공상적 사회주의", "en": "Utopian Socialism"},
        "original": "utopischer Sozialismus",
        "period": {"ko": "1800년대–1840년대", "en": "1800s–1840s"},
        "startYear": 1800,
        "endYear": 1848,
        "category": "theory",
        "parentId": UTOPIAN_SCIENTIFIC["id"],
        "aliases": {
            "ko": ["공상적 사회주의", "유토피아 사회주의", "공상적 사회주의자"],
            "en": ["utopian socialism", "utopian socialists"],
        },
        "people": ["friedrich-engels", "etienne-cabet"],
        "definition": {
            "ko": (
                "생시몽·푸리에·오언으로 대표되는 19세기 전반의 사회주의를, 엥겔스가 『공상에서 "
                "과학으로』에서 부른 이름. 세 사람은 특정 계급이 아니라 인류 전체를 해방하겠다고 나섰고, "
                "더 완전한 사회 질서를 머릿속에서 만들어 선전과 모범 실험으로 밖에서 사회에 부과하려 "
                "했다. 엥겔스는 이것을 개인의 결함이 아니라 자본주의와 계급 대립이 아직 미숙했던 시대 "
                "조건의 반영으로 설명하며, 그 환상적 외피를 뚫고 나온 통찰들 — 정치의 경제로의 흡수, "
                "여성 해방을 일반 해방의 척도로 본 시각 — 을 함께 기록했다."
            ),
            "en": (
                "Engels's name, in Socialism: Utopian and Scientific, for the socialism of the "
                "first half of the nineteenth century as represented by Saint-Simon, Fourier and "
                "Owen. None of the three came forward for a particular class: they set out to "
                "emancipate all humanity, working out a more perfect social order in the head and "
                "imposing it on society from outside by propaganda and model experiments. Engels "
                "explains this as a reflection of immature capitalist and class conditions rather "
                "than as a personal failing, and records the insights that broke through the "
                "fantastic covering — the absorption of politics by economics, and the degree of "
                "women's emancipation as the measure of general emancipation."
            ),
        },
        "body": {
            "ko": """## 이름이 뜻하는 것

'공상적'은 실현 불가능하다는 뜻이 아니라 근거의 자리를 가리키는 말이다. 해법을 역사 운동 안에서 찾지 않고 이성으로 설계해 밖에서 사회에 부과한다는 것. 엥겔스의 서술에서 이것은 조롱이 아니라 시대 진단이다. 1800년 무렵 자본주의의 모순은 이제 막 모습을 갖추기 시작했고, 그 모순을 끝낼 수단은 더더욱 그러했다. 프롤레타리아트는 스스로 돕지 못하는 억압받는 계층으로 나타났으므로, 도움은 밖에서 또는 위에서 오는 수밖에 없어 보였다.

## 세 사람

**생시몽(1760–1825).** 프랑스 혁명을 귀족과 부르주아지만이 아니라 무산자까지 셋이 벌인 계급 전쟁으로 읽었다. 1802년에 그것을 알아본 것은 엥겔스가 "가장 잉태력 있는 발견"이라 부른 통찰이다. 1816년에는 정치가 생산에 관한 과학이라고 선언하며 정치가 경제에 완전히 흡수될 것을 예고했다. 다만 그의 사회에서 지휘봉은 학자와 산업가 — 제조업자, 상인, 은행가 — 에게 있었고, 은행가가 신용 조절로 사회적 생산 전체를 이끌게 되어 있었다.

**푸리에(1772–1837).** 부르주아 세계의 물질적·도덕적 비참을 사정없이 드러낸 풍자가. 어떤 사회에서든 여성 해방의 정도가 일반적 해방의 자연스러운 척도라고 선언한 첫 사람이다. 역사를 야만·미개·가부장제·문명의 네 단계로 나누고, 문명이 스스로 만들어 낸 모순 속을 악순환하며 "과잉 자체에서 빈곤이 태어난다"고 썼다. 엥겔스는 그가 동시대인 헤겔만큼이나 능란하게 변증법을 구사했다고 평가한다.

**오언(1771–1858).** 뉴래너크 방적공장의 경영자였다. 1800년부터 1829년까지 2,500명의 마을을 술과 경찰과 소송과 구빈법이 사라진 모범 식민지로 바꿨다. 두 살부터 다니는 유아학교를 처음 세웠고, 경쟁자들이 하루 13~14시간을 시킬 때 10시간 반만 일을 시켰으며, 면화 공황으로 넉 달간 공장이 멈췄을 때도 임금을 전액 지급했다. 그러면서 사업은 두 배 넘게 커졌다. 그런데도 그는 만족하지 않았다. "사람들은 내 자비에 매인 노예였다."

## 무엇이 부족했나

엥겔스의 결론은 세 사람의 결함 목록이 아니라 한 문장으로 정리된다. 그들은 사회의 잘못을 제거해야 할 이성의 과제로 보았지 역사적으로 형성되어 스스로를 넘어설 수밖에 없는 생산양식의 문제로 보지 않았다. 그래서 그 체계들은 상세해질수록 순수한 환상으로 흘러갔다.

그리고 그 뒤에 오는 문장이 이 항목의 핵심이다. "우리는 그 환상적 외피를 뚫고 도처에서 터져 나오는 놀랍도록 위대한 사상과 그 맹아를 기뻐하며, 속물들은 거기에 눈이 멀어 있다."

## 왜 지금도 이 항목이 필요한가

'공상적'이라는 딱지는 20세기 내내 논쟁 상대를 밀어내는 도구로 쓰였다. 그러나 엥겔스의 용법에서 이 말은 상대를 논쟁에서 지우는 낱말이 아니라, 어떤 구상이 어떤 조건에서 나올 수밖에 없었는지를 묻는 낱말이다. 오늘 협동조합·공동체 실험·기본소득 설계에 이 딱지를 붙이려는 사람은 먼저 엥겔스가 오언의 유아학교와 노동시간 단축을 어떻게 기록했는지를 읽어야 한다.""",
            "en": """## What the Name Means

'Utopian' does not mean unrealisable; it names where the reasoning stands. The solution is not sought inside the historical movement but designed by reason and imposed on society from outside. In Engels's account this is a diagnosis of a period, not a jeer. Around 1800 the contradictions of capitalism were only beginning to take shape, and the means of ending them still more so. The proletariat appeared as an oppressed and suffering order incapable of helping itself, so help could seemingly come only from outside or from above.

## The Three

**Saint-Simon (1760–1825).** He read the French Revolution as a class war not merely between nobility and bourgeoisie but among three parties, the non-possessors included. To see that in 1802 was, in Engels's words, a most pregnant discovery. In 1816 he declared politics the science of production and foretold its complete absorption by economics. In his society, however, command lay with scholars and industrialists — manufacturers, merchants, bankers — and the bankers were to direct the whole of social production by regulating credit.

**Fourier (1772–1837).** A satirist who laid bare the material and moral misery of the bourgeois world, and the first to declare that in any given society the degree of woman's emancipation is the natural measure of general emancipation. He divided history into savagery, barbarism, the patriarchate and civilisation, and showed civilisation moving in a vicious circle of contradictions it reproduces without solving, so that 'under civilization poverty is born of superabundance itself'. Engels credits him with using dialectics as masterfully as his contemporary Hegel.

**Owen (1771–1858).** A cotton manufacturer. From 1800 to 1829 he turned New Lanark, a village of 2,500, into a model colony where drunkenness, police, magistrates, lawsuits and poor laws were unknown. He founded the first infant schools, taking children from the age of two; where competitors worked people thirteen or fourteen hours a day, New Lanark's day was ten and a half; when a cotton crisis stopped work for four months, his workers were paid in full. The business more than doubled in value. He was still not satisfied: 'The people were slaves at my mercy.'

## What Was Missing

Engels's conclusion is not a list of faults but a single point. They treated society's wrongs as a task for reason to remove, not as a problem of a historically formed mode of production bound to pass beyond itself. The more completely their systems were worked out in detail, the more they drifted into pure fantasy.

And then comes the sentence that matters most here: 'we delight in the stupendously grand thoughts and germs of thought that everywhere break out through their phantastic covering, and to which these Philistines are blind.'

## Why the Entry Still Earns Its Place

The label 'utopian' served through the twentieth century as a device for pushing opponents out of an argument. In Engels's usage it does no such work: it asks what conditions a given project could have arisen from. Anyone reaching for the label today — for co-operatives, community experiments, basic-income designs — should first read how Engels recorded Owen's infant schools and his shortened working day.""",
        },
    },
}

# ── 7. 과학적 사회주의 ────────────────────────────────────────────────

SCIENTIFIC = {
    "id": "scientific-socialism",
    "sources": [SRC_SUS_2, SRC_SUS_INDEX, SRC_SCIENTIFIC_COMMUNISM, SRC_SUS_3],
    "patch": {
        "term": {"ko": "과학적 사회주의", "en": "Scientific Socialism"},
        "original": "wissenschaftlicher Sozialismus",
        "period": {"ko": "1880년~", "en": "1880–present"},
        "startYear": 1880,
        "category": "theory",
        "parentId": UTOPIAN_SCIENTIFIC["id"],
        "aliases": {
            "ko": ["과학적 사회주의", "과학적 공산주의"],
            "en": ["scientific socialism", "scientific communism"],
        },
        "people": ["friedrich-engels", "karl-marx", "lenin", "suslov"],
        "definition": {
            "ko": (
                "사회주의를 도덕적 요청이 아니라 자본주의의 운동 법칙에 대한 분석에서 끌어내는 입장. "
                "엥겔스는 『공상에서 과학으로』에서 그 전환점을 두 발견으로 못박았다. 유물사관과, 잉여가치를 "
                "통해 밝혀진 자본주의적 생산의 비밀. \"이 두 위대한 발견과 함께 사회주의는 과학이 "
                "되었다.\" 이후 이 이름은 마르크스주의 조류의 자기 규정이 되었고, 소련에서는 1963년부터 "
                "'과학적 공산주의'라는 이름의 필수 교과가 되어 대학 교육에 편입되었다. 과학이라는 주장은 "
                "권위가 아니라 반증 가능성을 요구한다는 점에서 이 항목은 계속 논쟁 대상이다."
            ),
            "en": (
                "The position that socialism is derived from an analysis of the laws of motion of "
                "capitalism rather than from moral demand. In Socialism: Utopian and Scientific "
                "Engels fixed the turning point in two discoveries — the materialist conception of "
                "history, and the secret of capitalist production laid open through surplus value: "
                "'With these discoveries, Socialism became a science.' The name became the "
                "self-description of the Marxist current, and in the USSR it entered the "
                "university curriculum from 1963 as a compulsory subject called 'scientific "
                "communism'. Because a claim to be scientific demands falsifiability rather than "
                "authority, the entry remains contested ground."
            ),
        },
        "body": {
            "ko": """## 두 개의 발견

엥겔스의 정의는 짧다. 유물사관은 생산과 교환의 방식이 그때그때의 사회 구조와 정치·법·관념의 토대이며, 계급 대립의 해결책을 사람들의 머리가 아니라 그 시대의 생산양식에서 찾아야 한다고 본다. 잉여가치론은 자본주의적 취득의 비밀 — 자본가가 지불하지 않은 노동을 통해 부가 어떻게 만들어지는지 — 을 밝힌다. "이 두 위대한 발견을 우리는 마르크스에게 빚지고 있다. 이 발견들과 함께 사회주의는 과학이 되었다. 다음 일은 그 모든 세부와 연관을 밝혀내는 것이었다."

핵심은 '더 나은 사회'를 그리는 능력이 아니라 지금 사회의 운동 법칙에서 그 이행의 조건을 읽어내는 방법이다. 그래서 이 이름은 공상적 사회주의와 짝을 이룰 때에만 뜻이 분명해진다. 한쪽은 이성이 설계하고, 다른 쪽은 역사가 준비한 것을 읽는다.

## 이름의 내력

'과학적'이라는 수식은 엥겔스의 발명이 아니다. 19세기 프랑스 사회주의 논쟁에서 이미 돌아다니던 말이다. 다만 이 짝을 굳혀 널리 퍼뜨린 것이 1880년의 이 소책자이고, 그 뒤로 마르크스주의 조류는 스스로를 이 이름으로 불렀다.

## 교과목이 된 이름

소련에서 이 이름은 제도가 되었다. 1963년 6월 27일 명령 제214호로 '과학적 공산주의'가 대학 필수 교과로 도입되었다. 미하일 수슬로프가 주도한 조치였고, 역사유물론의 사회학적 부분을 흡수하는 형태였다. 마르크스와 엥겔스가 자기 관점을 가리키던 말이 학과 이름이 되고 시험 과목이 된 것이다.

이 이행은 양가적이다. 한편으로 그것은 이론이 국가 규모의 제도적 뒷받침을 얻었다는 뜻이고, 다른 한편으로 검증되어야 할 명제들이 교과서 항목으로 굳었다는 뜻이다. 후자의 대가는 1980년대에 분명해졌다.

## 무엇이 걸려 있는가

과학이라는 주장은 무거운 요구를 스스로 진다. 예측이 틀렸을 때 이론을 고칠 준비, 반대 증거를 회피하지 않을 태도가 그것이다. 마르크스와 엥겔스 자신은 그 요구에 자주 응했다. 『공산당 선언』 1872년 독일어판 서문은 파리 코뮌의 경험에 비추어 강령의 일부가 낡았다고 적고, 엥겔스는 이 소책자에서도 주식회사와 국가 인수 같은 새 형태를 이론에 편입시킨다.

거꾸로, '과학적'이라는 낱말이 결론을 미리 보증하는 인장으로 쓰일 때 이 항목은 자기 이름을 배반한다. 국가가 소유하면 사회주의라는 등식을 '과학적'이라 불러 온 역사가 그 예다. 엥겔스 자신이 같은 소책자의 각주에서 그 등식을 '사이비 사회주의'라 부른 것을 기억하면, 이 이름을 지키는 방법은 권위가 아니라 검증이라는 점이 분명해진다.""",
            "en": """## Two Discoveries

Engels's definition is short. The materialist conception of history holds that the mode of production and exchange is the basis of a society's structure and of its politics, law and ideas, and that the means of ending class antagonism are to be found in the mode of production of the epoch rather than in people's heads. The theory of surplus value lays open the secret of capitalist appropriation — how wealth is produced out of labour the capitalist does not pay for. 'These two great discoveries… we owe to Marx. With these discoveries, Socialism became a science. The next thing was to work out all its details and relations.'

The point is not a capacity to picture a better society but a method for reading the conditions of transition out of the laws of motion of this one. The name is only fully legible beside its pair: one side designs by reason, the other reads what history has prepared.

## Where the Name Comes From

The adjective was not Engels's invention; it was already circulating in nineteenth-century French socialist argument. What the 1880 pamphlet did was fix the pairing and carry it everywhere, after which the Marxist current used the name of itself.

## The Name Becomes a Course

In the USSR the name became an institution. On 27 June 1963, order no. 214 introduced 'scientific communism' as a compulsory subject in higher education. The move was driven by Mikhail Suslov, and the new subject absorbed the sociological part of historical materialism. A phrase Marx and Engels had used for their own standpoint became a department and an examination.

The shift cuts both ways. It meant the theory had institutional backing on the scale of a state; it also meant that propositions requiring testing hardened into textbook entries. The price of the second became plain in the 1980s.

## What Is at Stake

A claim to be scientific carries a heavy demand: a readiness to revise when predictions fail, and a refusal to duck contrary evidence. Marx and Engels often met it. The 1872 German preface to the Communist Manifesto records that parts of the programme had been overtaken by the experience of the Paris Commune, and in this pamphlet Engels folds new forms — joint-stock companies, state takeovers — into the analysis.

Conversely, the entry betrays its own name whenever 'scientific' is used as a seal guaranteeing conclusions in advance. The history of calling the equation of state ownership with socialism 'scientific' is the standing example. Remembering that Engels, in a footnote to this very pamphlet, called that equation a spurious socialism makes the point clear enough: the way to keep the name is verification, not authority.""",
        },
    },
}

# ── 8. 국가의 사멸 ────────────────────────────────────────────────────

WITHERING = {
    "id": "withering-away-of-the-state",
    "sources": [SRC_SUS_3, SRC_LENIN_1, SRC_LENIN_5, SRC_STALIN_18TH, SRC_GOTHA_4],
    "patch": {
        "term": {"ko": "국가의 사멸", "en": "Withering Away of the State"},
        "original": "Absterben des Staates",
        "period": {"ko": "1878년~", "en": "1878–present"},
        "startYear": 1878,
        "category": "theory",
        "parentId": UTOPIAN_SCIENTIFIC["id"],
        "aliases": {
            "ko": ["국가의 사멸", "국가 사멸"],
            "en": ["withering away of the state", "dying out of the state", "Absterben des Staates"],
        },
        "people": ["friedrich-engels", "lenin", "stalin", "karl-marx"],
        "definition": {
            "ko": (
                "엥겔스가 『반뒤링론』과 『공상에서 과학으로』에서 내놓은 명제. 국가는 계급 지배를 위한 "
                "특수한 억압 기구이므로, 억압할 계급이 사라지면 억압 기구도 필요를 잃는다. 사회 전체의 "
                "이름으로 생산수단을 장악하는 첫 행위가 국가로서의 마지막 독자적 행위이며, 그 뒤 사람에 "
                "대한 통치는 사물의 관리로 대체된다. \"국가는 '폐지'되는 것이 아니다. 그것은 사멸한다.\" "
                "레닌은 사멸하는 것이 혁명 뒤의 프롤레타리아 반(半)국가이고 부르주아 국가는 혁명으로 "
                "폐지된다고 정정했으며, 스탈린은 1939년 자본주의 포위가 남아 있는 한 국가는 존속한다고 "
                "선언해 이 명제를 사실상 뒤집었다."
            ),
            "en": (
                "The proposition Engels advanced in Anti-Dühring and in Socialism: Utopian and "
                "Scientific. The state is a special repressive force for class rule, so when there "
                "is no class left to hold down, the repressive force loses its function. The first "
                "act by which the state takes possession of the means of production in the name of "
                "society is at the same time its last independent act as a state, after which the "
                "government of persons gives way to the administration of things. 'The State is not "
                "\"abolished\". It dies out.' Lenin corrected the usual gloss — what withers away "
                "is the post-revolutionary proletarian semi-state, while the bourgeois state is "
                "abolished by revolution — and Stalin in 1939 effectively reversed the proposition, "
                "holding that the state remains so long as capitalist encirclement remains."
            ),
        },
        "body": {
            "ko": """## 명제의 자리

엥겔스의 논증은 국가의 정의에서 출발한다. 국가는 사회 전체를 대표하는 조직인 척하지만 실제로는 그때그때의 착취 계급의 조직이며, 무엇보다 피착취 계급을 억눌러 두기 위한 특수한 억압력이다. 고대에는 노예 소유 시민의 국가, 중세에는 봉건영주의 국가, 근대에는 부르주아지의 국가였다.

그렇다면 국가가 마침내 사회 전체의 진짜 대표가 되는 순간 국가는 스스로를 불필요하게 만든다. 억눌러야 할 계급이 없어지고, 생산의 무정부성에서 나오던 충돌과 과잉이 제거되면 억압할 것이 남지 않는다. 사회의 이름으로 생산수단을 장악하는 첫 행위는 국가로서의 마지막 독자적 행위다. 이후 사회관계에 대한 국가의 개입은 영역마다 차례로 불필요해지고 저절로 잠들어 간다. "사람에 대한 통치는 사물의 관리와 생산 과정의 지도로 대체된다. 국가는 '폐지'되는 것이 아니다. 그것은 사멸한다."

## 양쪽을 겨눈 명제

이 문장은 흔히 무정부주의자를 겨눈 것으로만 인용된다. 엥겔스는 실제로 국가를 당장 폐지하라는 요구를 이 명제로 반박했다. 그러나 같은 문단은 반대쪽도 겨눈다. 그는 곧바로 "자유로운 국가"라는 구호의 값이 이로써 드러난다고 덧붙인다. 선동가가 한때 쓸 수는 있으나 궁극적으로는 과학적으로 불충분한 말이라는 것이다. 마르크스도 『고타강령 비판』 4부에서 같은 구호를 두들겼다.

레닌은 1917년에 이 균형이 무너진 채 전해져 왔다고 지적했다. 무정부주의자를 겨눈 결론은 수천 번 되풀이되었지만 기회주의자를 겨눈 결론은 잊혔다는 것이다.

## 레닌의 정정

『국가와 혁명』 1장은 이 명제의 오독을 정면으로 다룬다. 엥겔스는 프롤레타리아트가 국가권력을 장악하면서 "국가를 국가로서 폐기한다"고 먼저 말했다. 곧 부르주아 국가는 사멸하는 것이 아니라 혁명으로 폐지된다. 사멸하는 것은 그 뒤에 남은 프롤레타리아 국가, 또는 반(半)국가다. 하나의 특수한 억압력을 다른 특수한 억압력으로 갈아 끼우는 일은 결코 '사멸'의 형식으로 일어날 수 없다.

이 정정이 겨눈 것은 사멸 명제를 혁명 불필요론의 근거로 쓰는 독법이었다. 국가가 어차피 사멸한다면 지금 국가를 깨뜨릴 이유가 없다는 논법 말이다.

## 1939년의 역전

소련에서 이 명제는 다른 방향에서 무너진다. 1939년 3월 18차 당대회 보고에서 스탈린은 사회주의가 승리한 뒤에도 국가가 남는가를 묻고 이렇게 답했다. 자본주의 포위가 청산되지 않고 외부의 군사 공격 위험이 사라지지 않는 한 국가는 남는다. 엥겔스의 공식은 사회주의가 모든 나라 또는 대다수 나라에서 승리한 경우를 전제한 것이며, 한 나라에서만 승리한 경우에 대해 고전들은 답을 줄 수 없었다는 것이다.

형식상 이것은 이론의 구체화다. 실제 효과는 사멸의 무기한 연기였다. 국가기구는 사멸을 향해 축소되기는커녕 같은 시기 최대로 팽창했고, 이론은 그 팽창을 설명하는 도구가 되었다.

## 오늘 읽는 법

세 층을 갈라 읽어야 한다. 엥겔스의 명제는 국가의 소멸을 예언한 것이 아니라 국가의 성격을 규정한 것이다. 국가가 계급 지배의 기구인 한, 계급이 사라지면 그 기구도 존재 근거를 잃는다는 조건문이다. 레닌의 정정은 그 조건문이 혁명을 면제해 주지 않는다는 경고다. 그리고 1939년의 답변은 조건이 충족되지 않았을 때 이론이 어떻게 현실을 추인하는 쪽으로 굽는지를 보여 주는 사례다.

이 세 층을 겹쳐 놓으면 국가사회주의라는 말이 왜 분석 도구가 되지 못했는지가 다시 보인다. 국가가 무엇을 소유하는가는 국가의 성격을 바꾸지 못한다. 물어야 할 것은 그 국가가 누구의 것인가, 그리고 그것이 자기를 불필요하게 만드는 방향으로 가고 있는가이다.""",
            "en": """## Where the Proposition Sits

Engels's argument starts from a definition. The state pretends to be the organisation of society as a whole but is in fact the organisation of whichever class is for the time being the exploiting one, and above all a special repressive force for holding the exploited class down — the state of the slave-owning citizens in antiquity, of the feudal lords in the Middle Ages, of the bourgeoisie in modern times.

It follows that at the moment the state becomes the real representative of the whole of society, it renders itself unnecessary. With no class left to hold in subjection, and with the collisions and excesses of anarchy in production removed, nothing remains to be repressed. The first act by which the state takes possession of the means of production in the name of society is at the same time its last independent act as a state. Thereafter state interference in social relations becomes superfluous in one domain after another and dies out of itself: 'the government of persons is replaced by the administration of things, and by the conduct of processes of production. The State is not "abolished". It dies out.'

## Aimed at Both Sides

The sentence is usually quoted as though it were aimed only at the anarchists, and Engels did use it against the demand for abolition of the state out of hand. But the same paragraph turns the other way as well: it gives, he adds, the measure of the value of the phrase 'a free State', both as to its justifiable use at times by agitators and as to its ultimate scientific insufficiency. Marx hammered the same slogan in Part IV of the Critique of the Gotha Programme.

In 1917 Lenin pointed out that the balance had not survived transmission: the conclusion against the anarchists had been repeated thousands of times, while the conclusion against the opportunists was forgotten.

## Lenin's Correction

Chapter 1 of The State and Revolution takes the misreading head-on. Engels says first that in seizing state power the proletariat 'abolishes the state as state' — that is, the bourgeois state does not wither, it is abolished by the revolution. What withers away afterwards is the proletarian state, or semi-state. Replacing one special repressive force with another cannot possibly take the form of withering.

The target of the correction was the reading that turned the proposition into an argument against revolution: if the state withers away in any case, why break it now?

## The Reversal of 1939

In the USSR the proposition failed from the other direction. In his report to the 18th Party Congress in March 1939, Stalin asked whether the state would remain under communism and answered that it would, unless capitalist encirclement were liquidated and the danger of foreign military attack had disappeared. Engels's formula, he argued, assumed socialism victorious in all or most countries; for socialism victorious in one country alone the classics 'could not have given an answer'.

Formally this is a concretisation of theory. Its effect was to postpone the withering indefinitely. Far from shrinking toward its own disappearance, the state apparatus reached its greatest extent in exactly those years, and the theory became an instrument for explaining that growth.

## How to Read It Now

Three layers, kept apart. Engels's proposition is not a prophecy of the state's disappearance but a characterisation of what a state is: a conditional statement that an instrument of class rule loses its ground when classes go. Lenin's correction is a warning that the conditional grants no exemption from revolution. And the answer of 1939 is a case study in how a theory bends toward ratifying the present when the condition has not been met.

Lay the three over one another and it becomes visible again why 'state socialism' never worked as a tool of analysis. What the state owns does not change what the state is. The questions are whose state it is, and whether it is moving toward making itself unnecessary.""",
        },
    },
}

TERMS = [
    GOTHA, PHASES, PROCEEDS, IRON_LAW,
    UTOPIAN_SCIENTIFIC, UTOPIAN, SCIENTIFIC, WITHERING,
]


def report(entry: dict) -> list[str]:
    problems: list[str] = []
    patch = entry["patch"]
    definition, body = patch["definition"], patch["body"]
    parent = patch.get("parentId") or "—"
    print(f"\nterm: {entry['id']}")
    print(f"  headword      {patch['term']['ko']} / {patch['term']['en']}")
    print(f"  category      {patch['category']}   period {patch['period']['ko']}   parent {parent}")
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
        "--sources-only", action="store_true",
        help="update the sources column of entries that already exist",
    )
    parser.add_argument(
        "--update", action="store_true",
        help="rewrite entries that already exist, sending the whole patch",
    )
    args = parser.parse_args()

    # commulingo_terms.sources is a patch field, not the tool-level citations
    # argument — passing citations alone records provenance on the suggestion row
    # and leaves the entry page's 출처 list empty. Carry both.
    for entry in TERMS:
        entry["patch"]["sources"] = entry["sources"]

    if args.sources_only:
        from runtime_tools.commulingo_people import _exec_commulingo_write

        failed = 0
        for entry in TERMS:
            if args.only and entry["id"] != args.only:
                continue
            if not args.apply:
                print(f"{entry['id']:<36} sources -> {len(entry['sources'])}")
                continue
            result = await _exec_commulingo_write(
                "term", "update", entry["id"], entry["sources"],
                {"sources": entry["sources"]}, 0.95,
            )
            print(f"{entry['id']}: {result}")
            if result.startswith("Error:") or '"error"' in result:
                failed += 1
        if not args.apply:
            print("\ndry run; pass --apply to write")
        return 1 if failed else 0

    entries = [t for t in TERMS if not args.only or t["id"] == args.only]
    problems: list[str] = []
    for entry in entries:
        problems.extend(report(entry))

    # Parents must exist before their children; the list is already ordered that
    # way, but a --only run can violate it, so say so rather than fail in the DB.
    ids = {t["id"] for t in TERMS}
    seen: set[str] = set()
    for entry in entries:
        parent = entry["patch"].get("parentId")
        if parent and parent in ids and parent not in seen and not args.only:
            problems.append(f"{entry['id']} is listed before its parent {parent}")
        seen.add(entry["id"])

    if problems:
        print("\nABORT:")
        for problem in problems:
            print(f"  {problem}")
        return 1

    if not args.apply:
        print("\ndry run; pass --apply to write")
        return 0

    from runtime_tools.commulingo_people import _exec_commulingo_write

    # The same patch serves both actions: term update accepts every key create
    # does, and re-sending the stored parentId is a no-op rather than an error.
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
