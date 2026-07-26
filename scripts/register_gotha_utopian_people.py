#!/usr/bin/env python3
"""Register the four German people the Gotha/Anti-Dühring entries stand on.

The people dictionary holds 930 cards and reaches back past Marx — Cabet
(b. 1788), Blanqui, Bakunin, Chernyshevsky are all in it — but Marx and Engels
themselves were never entered, nor were the two men the Gotha quarrel was
between. So the glossary entries on 『고타강령 비판』 and 『공상에서 과학으로』
had no author to link to, and the state-socialism entry had to name Wilhelm
Liebknecht in prose with an explicit note that he could not be linked because
the 'liebknecht' record is his son Karl.

Four cards close that hole:

  karl-marx            1818–1883  author of the Critique
  friedrich-engels     1820–1895  author of the pamphlet, publisher of the Critique
  ferdinand-lassalle   1825–1864  the target of both
  wilhelm-liebknecht   1826–1900  drafter of the programme Marx dissected

Wilhelm, not Karl. The 'liebknecht' slug stays with the son; the duplicate
guard lets 'wilhelm-liebknecht' through because the birth years differ, and
the linkifier will stop auto-linking the bare surname 리프크네히트 to either
man once two people own the word — which is the correct outcome, not a
regression.

Usage:
  venv/bin/python scripts/register_gotha_utopian_people.py            # dry run
  venv/bin/python scripts/register_gotha_utopian_people.py --apply
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

GERMANY = {"code": "germany", "label": {"ko": "독일", "en": "Germany"}}

MARX = {
    "id": "karl-marx",
    "sources": [
        "https://en.wikipedia.org/wiki/Karl_Marx — life dates (Trier 1818 – London 1883), the Rheinische Zeitung editorship and its suppression, the 1845 renunciation of Prussian citizenship and lifelong statelessness, the London exile from 1849, and the leadership of the International Working Men's Association 1864–1872",
        "https://www.marxists.org/archive/marx/works/1875/gotha/index.htm — Marx/Engels Selected Works: the Critique of the Gotha Programme was 'written April or early May, 1875' and sent to Bracke for circulation among the party leadership; first published, abridged, in Die Neue Zeit in 1891",
        "https://www.marxists.org/archive/marx/works/1875/gotha/ch01.htm — the closing sentence of Part I: 'only then can the narrow horizon of bourgeois right be crossed in its entirety and society inscribe on its banners: From each according to his ability, to each according to his needs!'",
    ],
    "patch": {
        "group": "international-revolutionary",
        "cyrillic": "Karl Marx",
        "years": "1818–1883",
        "givenName": {"ko": "카를", "en": "Karl"},
        "familyName": {"ko": "마르크스", "en": "Marx"},
        "epithet": {
            "ko": "『자본론』의 저자, 제1인터내셔널을 이끈 사람",
            "en": "Author of Capital and leader of the First International",
        },
        "bio": {
            "ko": (
                "트리어의 법률가 집안에서 태어나 『라인신문』 편집장으로 출발했다. 1845년 프로이센 "
                "국적을 버린 뒤 죽을 때까지 무국적자였고, 1849년부터는 런던 망명자였다. 엥겔스와 "
                "40년을 함께 일하며 『공산당 선언』과 『자본론』을 썼고 제1인터내셔널을 이끌었다. "
                "만년의 『고타강령 비판』(1875)은 독일 노동자 정당의 통합강령에 들어앉은 라살레주의를 "
                "조목조목 해부한 문서로, 국가를 사회주의의 도구로 보는 발상에 대한 그의 마지막 답변이다."
            ),
            "en": (
                "Born into a lawyer's family in Trier, he began as editor of the Rheinische "
                "Zeitung. He renounced his Prussian citizenship in 1845 and remained stateless "
                "for the rest of his life, living in London exile from 1849. Over forty years "
                "with Engels he wrote the Communist Manifesto and Capital and led the First "
                "International. The late Critique of the Gotha Programme (1875) takes apart, "
                "line by line, the Lassallean thinking lodged in the German workers' party's "
                "unity programme — his last answer to the idea of the state as the instrument "
                "of socialism."
            ),
        },
        "moment": {
            "ko": "「각자는 능력에 따라, 각자에게는 필요에 따라!」 — 『고타강령 비판』(1875)",
            "en": (
                "\"From each according to his ability, to each according to his needs!\" "
                "— Critique of the Gotha Programme (1875)"
            ),
        },
        "fate": {"kind": "natural", "label": {"ko": "자연사", "en": "Natural causes"}},
        "citizenship": GERMANY,
        "nationalOrigin": GERMANY,
        "role": {"categoryId": "theorist"},
        "aliases": {"ko": ["마르크스"], "en": ["Marx"]},
        "career": [
            {"y": "1842–1843", "r": {"ko": "『라인신문』 편집장 — 프로이센 검열로 폐간", "en": "Editor of the Rheinische Zeitung until Prussian censorship closed it"}},
            {"y": "1844", "r": {"ko": "파리에서 엥겔스와 만나 평생의 공동작업을 시작", "en": "Met Engels in Paris, beginning a lifelong collaboration"}},
            {"y": "1845", "r": {"ko": "프로이센 국적 포기 — 이후 죽을 때까지 무국적자", "en": "Renounced Prussian citizenship; stateless thereafter"}},
            {"y": "1848", "r": {"ko": "『공산당 선언』 발표, 『신라인신문』 편집장", "en": "Published the Communist Manifesto; edited the Neue Rheinische Zeitung"}},
            {"y": "1849–1883", "r": {"ko": "런던 망명 — 대영박물관 열람실에서 『자본론』을 쓰다", "en": "Exile in London, writing Capital in the British Museum reading room"}},
            {"y": "1864–1872", "r": {"ko": "국제노동자협회 창립선언문 집필, 총평의회 지도", "en": "Drafted the Inaugural Address of the International Working Men's Association and led its General Council"}},
            {"y": "1867", "r": {"ko": "『자본론』 제1권 출간", "en": "Published Capital, Volume I"}},
            {"y": "1871", "r": {"ko": "『프랑스 내전』 — 파리 코뮌을 노동계급 정부의 형태로 옹호", "en": "The Civil War in France — defended the Paris Commune as a form of working-class government"}},
            {"y": "1875", "r": {"ko": "『고타강령 비판』 집필 — 브라케에게 보내 지도부에만 회람", "en": "Wrote the Critique of the Gotha Programme, sent to Bracke for the leadership alone"}},
            {"y": "1883", "r": {"ko": "런던에서 사망, 하이게이트 묘지에 묻히다", "en": "Died in London and was buried at Highgate Cemetery"}},
        ],
    },
}

ENGELS = {
    "id": "friedrich-engels",
    "sources": [
        "https://en.wikipedia.org/wiki/Friedrich_Engels — life dates (Barmen 1820 – London 1895), the Manchester firm of Ermen and Engels, The Condition of the Working Class in England (1845), service as adjutant in the 1849 Baden-Palatinate campaign, the twenty years of clerking that supported the Marx household, and the editing of Capital vols. II and III after Marx's death",
        "https://www.marxists.org/archive/marx/works/1880/soc-utop/index.htm — Engels in the preface to the 1892 English edition: 'At the request of my friend, Paul Lafargue… I arranged three chapters of this book as a pamphlet, which he translated and published in 1880… In 1883, our German friends brought out the pamphlet in the original language… I am not aware that any other Socialist work, not even our Communist Manifesto of 1848, or Marx's Capital, has been so often translated.'",
        "https://www.marxists.org/archive/marx/works/1880/soc-utop/ch03.htm — 'The State is not \"abolished\". It dies out.'",
        "https://www.marxists.org/archive/marx/works/1875/gotha/foreword.htm — Engels's foreword of 6 January 1891 explaining why he published the manuscript over the party's objections after the Halle Congress put the Gotha Programme on the agenda, and what he cut: 'I have omitted a few sharp personal expressions and judgments… and replaced them by dots.'",
    ],
    "patch": {
        "group": "international-revolutionary",
        "cyrillic": "Friedrich Engels",
        "years": "1820–1895",
        "givenName": {"ko": "프리드리히", "en": "Friedrich"},
        "familyName": {"ko": "엥겔스", "en": "Engels"},
        "epithet": {
            "ko": "마르크스의 공동저자이자 후원자, 『공상에서 과학으로』의 저자",
            "en": "Marx's co-author and patron; author of Socialism: Utopian and Scientific",
        },
        "bio": {
            "ko": (
                "바르멘의 방적공장주 집안에서 태어나 맨체스터의 가업을 관리하며 『영국 노동계급의 "
                "상태』(1845)를 썼다. 1848년 혁명과 바덴 봉기를 거친 뒤 20년을 상사에 매여 런던의 "
                "마르크스 가족을 부양했다. 뒤링을 반박한 『반뒤링론』(1878)에서 세 개 장을 뽑아 만든 "
                "소책자가 『공상에서 과학으로』(1880)이고, 이 책은 곧 열 개 언어로 번역되어 마르크스주의 "
                "입문서가 되었다. 마르크스 사후에는 『자본론』 2·3권을 편집했고, 1891년 15년간 묻혀 "
                "있던 『고타강령 비판』을 당 지도부의 반대를 무릅쓰고 발표했다."
            ),
            "en": (
                "Born to a Barmen mill-owning family, he managed the family firm in Manchester "
                "and wrote The Condition of the Working Class in England (1845). After the 1848 "
                "revolution and the Baden rising he returned to the office for twenty years to "
                "keep the Marx household in London. From Anti-Dühring (1878) he drew three "
                "chapters into the pamphlet Socialism: Utopian and Scientific (1880), which was "
                "translated into ten languages and became the introduction to Marxism. After "
                "Marx's death he edited Capital vols. II and III, and in 1891 he published the "
                "Critique of the Gotha Programme — buried for fifteen years — over the "
                "objections of the party leadership."
            ),
        },
        "moment": {
            "ko": "「국가는 '폐지'되는 것이 아니다. 그것은 사멸한다.」 — 『공상에서 과학으로』(1880)",
            "en": "\"The State is not 'abolished'. It dies out.\" — Socialism: Utopian and Scientific (1880)",
        },
        "fate": {"kind": "natural", "label": {"ko": "자연사", "en": "Natural causes"}},
        "citizenship": GERMANY,
        "nationalOrigin": GERMANY,
        "role": {"categoryId": "theorist"},
        "aliases": {"ko": ["엥겔스"], "en": ["Engels"]},
        "career": [
            {"y": "1842–1844", "r": {"ko": "맨체스터 에르멘 운트 엥겔스 상사 근무 — 『영국 노동계급의 상태』의 현장", "en": "Clerked at Ermen and Engels in Manchester — the field for The Condition of the Working Class in England"}},
            {"y": "1844", "r": {"ko": "파리에서 마르크스와 만나 평생의 공동작업을 시작", "en": "Met Marx in Paris, beginning a lifelong collaboration"}},
            {"y": "1848", "r": {"ko": "『공산당 선언』 공동 집필, 『신라인신문』 편집진", "en": "Co-wrote the Communist Manifesto; on the staff of the Neue Rheinische Zeitung"}},
            {"y": "1849", "r": {"ko": "바덴-팔츠 봉기에 부관으로 참전", "en": "Fought in the Baden-Palatinate campaign as an adjutant"}},
            {"y": "1850–1870", "r": {"ko": "맨체스터 상사로 복귀 — 20년간 마르크스 가족을 부양", "en": "Returned to the Manchester firm and supported the Marx family for twenty years"}},
            {"y": "1878", "r": {"ko": "『반뒤링론』 출간 — 각주에서 비스마르크식 국유화를 '사이비 사회주의'라 부르다", "en": "Published Anti-Dühring, whose footnote calls Bismarckian nationalisation 'a kind of spurious Socialism'"}},
            {"y": "1880", "r": {"ko": "라파르그의 요청으로 세 개 장을 발췌해 『공상에서 과학으로』를 프랑스어로 발표", "en": "At Lafargue's request drew out three chapters as Socialisme utopique et Socialisme scientifique"}},
            {"y": "1883–1894", "r": {"ko": "마르크스 사후 『자본론』 제2권과 제3권 편집", "en": "Edited Capital vols. II and III from Marx's manuscripts"}},
            {"y": "1891", "r": {"ko": "『고타강령 비판』을 『신시대』에 발표 — 당의 반대를 서문으로 정면 돌파", "en": "Published the Critique of the Gotha Programme in Die Neue Zeit, answering the party's objections in his foreword"}},
            {"y": "1895", "r": {"ko": "런던에서 사망", "en": "Died in London"}},
        ],
    },
}

LASSALLE = {
    "id": "ferdinand-lassalle",
    "sources": [
        "https://en.wikipedia.org/wiki/Ferdinand_Lassalle — born Breslau 11 April 1825, died Carouge near Geneva 31 August 1864 of a wound taken in a duel with Yanko von Rakowitza on 28 August; founded the General German Workers' Association (ADAV) at Leipzig on 23 May 1863 and was elected president for a five-year term; his programme of state-financed producers' co-operatives achieved through universal suffrage; his secret meetings and correspondence with Bismarck from May 1863, which drew the deep suspicion of Marx",
        "https://en.wikipedia.org/wiki/Iron_law_of_wages — Lassalle's ehernes Lohngesetz, his phrase 'das eiserne und grausame Gesetz', and the borrowing of 'iron' from Goethe's 'great, eternal iron laws' in Das Göttliche",
        "https://www.marxists.org/archive/marx/works/1875/gotha/ch02.htm — Marx: 'It is well known that nothing of the \"iron law of wages\" is Lassalle's except the word \"iron\" borrowed from Goethe's \"great, eternal iron laws\".'",
    ],
    "patch": {
        "group": "international-revolutionary",
        "cyrillic": "Ferdinand Lassalle",
        "years": "1825–1864",
        "givenName": {"ko": "페르디난트", "en": "Ferdinand"},
        "familyName": {"ko": "라살레", "en": "Lassalle"},
        "epithet": {
            "ko": "독일 최초의 노동자 정당 ADAV의 창립자",
            "en": "Founder of the ADAV, Germany's first workers' party",
        },
        "bio": {
            "ko": (
                "브레슬라우의 유대인 상인 집안에서 태어난 변호사이자 선동가. 1863년 5월 라이프치히에서 "
                "전독일노동자협회(ADAV)를 세워 5년 임기의 회장이 되었고, 보통선거권으로 국가를 움직여 "
                "국가 보조 생산협동조합을 세우면 사회주의에 이른다고 주장했다. 같은 해 비스마르크와 "
                "비밀리에 만나 노동자와 프로이센 왕정의 동맹을 타진해 마르크스의 깊은 의심을 샀다. "
                "1864년 결혼 문제로 벌인 결투에서 총상을 입고 사흘 뒤 제네바 근교에서 죽었다. 그가 "
                "남긴 강령은 11년 뒤 고타 통합강령의 뼈대가 되었다."
            ),
            "en": (
                "A lawyer and agitator from a Jewish merchant family in Breslau. In May 1863 he "
                "founded the General German Workers' Association (ADAV) at Leipzig and was "
                "elected its president for five years, arguing that universal suffrage would "
                "turn the state toward funding producers' co-operatives and so toward socialism. "
                "That same year he met Bismarck in secret to sound out an alliance between the "
                "workers and the Prussian monarchy, which drew Marx's deep suspicion. In 1864 he "
                "was shot in a duel over a marriage and died three days later outside Geneva. "
                "Eleven years afterwards his programme supplied the frame of the Gotha unity "
                "programme."
            ),
        },
        "moment": {
            "ko": (
                "「철과 같은, 잔혹한 법칙(das eiserne und grausame Gesetz)」 — 임금을 생존 "
                "수준에 묶어 둔다며 라살레가 임금철칙에 붙인 이름(1863)"
            ),
            "en": (
                "\"Das eiserne und grausame Gesetz\" — the iron and cruel law Lassalle said held "
                "wages down to bare subsistence (1863)"
            ),
        },
        "fate": {"kind": "killed", "label": {"ko": "결투 중 총상", "en": "Shot in a duel"}},
        "citizenship": GERMANY,
        "nationalOrigin": GERMANY,
        "role": {"categoryId": "theorist"},
        "aliases": {"ko": ["라살레"], "en": ["Lassalle"]},
        "career": [
            {"y": "1848–1849", "r": {"ko": "라인란트에서 혁명 선동, 무력 저항 선동 혐의로 6개월 금고", "en": "Agitated in the Rhineland during the revolution; jailed six months for inciting armed resistance"}},
            {"y": "1862", "r": {"ko": "「노동자 강령」 강연 — 노동자 계급을 독자적 정치 세력으로 호명", "en": "The Workers' Programme lecture, addressing the working class as an independent political force"}},
            {"y": "1863.5", "r": {"ko": "라이프치히에서 전독일노동자협회(ADAV) 창립, 5년 임기 회장 취임", "en": "Founded the ADAV at Leipzig and became its president for a five-year term"}},
            {"y": "1863", "r": {"ko": "비스마르크와 비밀 회동·서신 — 노동자와 프로이센 왕정의 동맹을 타진", "en": "Secret meetings and correspondence with Bismarck on a workers–monarchy alliance"}},
            {"y": "1864.8", "r": {"ko": "제네바 근교 결투에서 총상, 사흘 뒤 사망", "en": "Shot in a duel outside Geneva; died three days later"}},
        ],
    },
}

WILHELM_LIEBKNECHT = {
    "id": "wilhelm-liebknecht",
    "sources": [
        "https://en.wikipedia.org/wiki/Wilhelm_Liebknecht — born Giessen 29 March 1826, died Charlottenburg 7 August 1900; the 1848 risings and twelve years of London exile as a close associate of Marx and Engels; co-founder with August Bebel of the Social Democratic Workers' Party at Eisenach in 1869; arrest in December 1870 for opposing war credits and the March 1872 Leipzig treason trial; principal architect of the 1875 Gotha unity congress and author of the Gotha Programme; editor-in-chief of Vorwärts after 1890; father of Karl Liebknecht (1871–1919)",
        "https://www.marxists.org/archive/liebknecht-w/1896/08/our-congress.htm — 'Our Recent Congress', Paris, 10 August 1896: 'Nobody has combatted State Socialism more than we German Socialists, nobody has shown more distinctively than I, that State Socialism is really State capitalism!'",
        "https://www.marxists.org/archive/marx/works/1875/gotha/foreword.htm — Engels's 1891 foreword confirms the manuscript was sent in 1875 'to Bracke for communication to Geib, Auer, Bebel, and Liebknecht and subsequent return to Marx' — this Liebknecht is Wilhelm",
    ],
    "patch": {
        "group": "international-revolutionary",
        "cyrillic": "Wilhelm Liebknecht",
        "years": "1826–1900",
        "givenName": {"ko": "빌헬름", "en": "Wilhelm"},
        "familyName": {"ko": "리프크네히트", "en": "Liebknecht"},
        "epithet": {
            "ko": "아이제나흐파 창립자, 고타 통합강령의 기초자",
            "en": "Co-founder of the Eisenach party and drafter of the Gotha Programme",
        },
        "bio": {
            "ko": (
                "기센에서 태어나 1848년 봉기에 가담했다가 스위스를 거쳐 런던으로 망명해 12년간 "
                "마르크스·엥겔스의 가까운 동지로 지냈다. 1869년 베벨과 함께 아이제나흐에서 "
                "사회민주노동자당(SDAP)을 세웠고, 보불전쟁 전쟁공채에 반대하다 체포되어 1872년 "
                "라이프치히 반역죄 재판에서 요새금고형을 받았다. 1875년 고타 통합대회를 성사시킨 "
                "주역이자 통합강령의 기초자였으며, 마르크스가 조목조목 해부한 문서가 바로 그것이다. "
                "1890년부터 죽을 때까지 당 기관지 『전진』을 편집했다. 카를 리프크네히트의 아버지다."
            ),
            "en": (
                "Born in Giessen, he joined the risings of 1848 and fled by way of Switzerland to "
                "London, where he spent twelve years as a close associate of Marx and Engels. In "
                "1869 he founded the Social Democratic Workers' Party with August Bebel at "
                "Eisenach; arrested for opposing war credits, he was convicted at the Leipzig "
                "treason trial of 1872. He was the man who brought off the Gotha unity congress "
                "of 1875 and drafted its programme — the document Marx took apart line by line. "
                "From 1890 until his death he edited the party paper Vorwärts. He was the father "
                "of Karl Liebknecht."
            ),
        },
        "moment": {
            "ko": (
                "「국가사회주의와 가장 앞장서 싸워 온 것이 우리 독일 사회주의자들이며, 국가사회주의가 "
                "실은 국가자본주의임을 누구보다 분명히 보인 사람이 나다.」 — '우리의 최근 대회'(1896)"
            ),
            "en": (
                "\"Nobody has combatted State Socialism more than we German Socialists, nobody has "
                "shown more distinctively than I, that State Socialism is really State "
                "capitalism!\" — 'Our Recent Congress' (1896)"
            ),
        },
        "fate": {"kind": "natural", "label": {"ko": "자연사", "en": "Natural causes"}},
        "citizenship": GERMANY,
        "nationalOrigin": GERMANY,
        "role": {"categoryId": "theorist"},
        "aliases": {"ko": ["빌헬름 리프크네히트"], "en": ["Wilhelm Liebknecht"]},
        "career": [
            {"y": "1848–1849", "r": {"ko": "바덴 봉기 참가, 패배 후 스위스로 망명", "en": "Fought in the Baden rising and fled to Switzerland after its defeat"}},
            {"y": "1850–1862", "r": {"ko": "런던 망명 — 마르크스·엥겔스와 교유하며 공산주의자동맹에 가담", "en": "Exile in London, close to Marx and Engels, and a member of the Communist League"}},
            {"y": "1869", "r": {"ko": "베벨과 아이제나흐에서 사회민주노동자당(SDAP) 창립", "en": "Co-founded the Social Democratic Workers' Party at Eisenach with Bebel"}},
            {"y": "1870–1872", "r": {"ko": "전쟁공채 반대로 체포, 라이프치히 반역죄 재판에서 2년 요새금고형", "en": "Arrested for opposing war credits; sentenced to two years' fortress confinement at the Leipzig treason trial"}},
            {"y": "1875", "r": {"ko": "고타 통합대회 주도 — 통합강령을 기초해 라살레파와 합당", "en": "Led the Gotha unity congress and drafted the programme that merged the party with the Lassalleans"}},
            {"y": "1878–1890", "r": {"ko": "사회주의자탄압법 아래 제국의회 의원으로 당을 지키다", "en": "Held the party together as a Reichstag deputy under the Anti-Socialist Laws"}},
            {"y": "1890–1900", "r": {"ko": "당 기관지 『전진(Vorwärts)』 편집장", "en": "Editor-in-chief of the party paper Vorwärts"}},
            {"y": "1900", "r": {"ko": "샤를로텐부르크에서 사망", "en": "Died in Charlottenburg"}},
        ],
    },
}

PEOPLE = [MARX, ENGELS, LASSALLE, WILHELM_LIEBKNECHT]

LIMITS = (
    ("epithet", 60, 140),
    ("bio", 380, 900),
    ("moment", 140, 300),
)


def report(entry: dict) -> list[str]:
    problems: list[str] = []
    patch = entry["patch"]
    print(f"\nperson: {entry['id']}  ({patch['years']})")
    print(f"  name          {patch['givenName']['ko']} {patch['familyName']['ko']} / "
          f"{patch['givenName']['en']} {patch['familyName']['en']}")
    print(f"  native        {patch['cyrillic']}")
    for key, ko_max, en_max in LIMITS:
        value = patch.get(key) or {}
        ko, en = len(value.get("ko") or ""), len(value.get("en") or "")
        flag = "  OVER" if (ko > ko_max or en > en_max) else ""
        print(f"  {key:<13} ko {ko:>3}/{ko_max}   en {en:>3}/{en_max}{flag}")
        if flag:
            problems.append(f"{entry['id']}.{key} exceeds the limit")
    label = patch["fate"]["label"]
    if len(label["ko"]) > 22 or len(label["en"]) > 50:
        problems.append(f"{entry['id']}.fate.label exceeds the limit")
    print(f"  fate          {patch['fate']['kind']} / {label['ko']} ({len(label['ko'])}/22)")
    print(f"  career        {len(patch['career'])} rows")
    print(f"  sources       {len(entry['sources'])}")
    return problems


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--only", help="register a single id")
    parser.add_argument(
        "--update", action="store_true",
        help="rewrite cards that already exist, sending the whole patch",
    )
    args = parser.parse_args()

    entries = [p for p in PEOPLE if not args.only or p["id"] == args.only]
    problems: list[str] = []
    for entry in entries:
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

    # A person update takes the same patch a create does, so re-sending the whole
    # card is the way to correct one field without hand-writing a partial patch.
    action = "update" if args.update else "create"
    failed = 0
    for entry in entries:
        result = await _exec_commulingo_write(
            "person", action, entry["id"], entry["sources"], entry["patch"], 0.95,
        )
        print(f"\n{entry['id']}: {result}")
        if result.startswith("Error:") or '"error"' in result:
            failed += 1
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
