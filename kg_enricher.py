"""
kg_enricher.py — KG 엔티티 설명 자동 생성
MOON PC llama-server (qwen3.5-9b) → 로컬 Docker Neo4j
폴백: 로컬 Ollama (qwen3.5:4b)
"""

import json
import argparse
import os
from datetime import datetime
from neo4j import GraphDatabase
from ollama_client import ask

# 로컬 Docker Neo4j
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

# ─── LLM ────────────────────────────────────────────────
def query_llm(name: str, labels: list) -> str:
    label_str = ", ".join(l for l in labels if l != "Entity")
    prompt = (
        f"엔티티 이름: {name}\n"
        f"유형: {label_str if label_str else '일반'}\n\n"
        f"위 엔티티에 대해 2~3문장으로 간결하게 설명해줘. "
        f"지정학, 정치경제, 역사 맥락을 중심으로. "
        f"불필요한 서론 없이 바로 설명만 써줘."
    )
    return ask(prompt, temperature=0.4).strip()

# ─── Neo4j ──────────────────────────────────────────────
def get_entities(driver, limit=None):
    """summary 없는 엔티티 조회 (US 중복 제외)"""
    query = """
    MATCH (e:Entity)
    WHERE (e.summary IS NULL OR e.summary = "")
      AND e.name <> "US"
    RETURN e.name AS name, labels(e) AS labels, elementId(e) AS eid
    ORDER BY e.name
    """
    if limit:
        query += f" LIMIT {limit}"
    with driver.session() as session:
        return [dict(r) for r in session.run(query)]

def update_summary(driver, eid: str, summary: str):
    """Neo4j elementId로 summary 업데이트"""
    with driver.session() as session:
        session.run(
            "MATCH (e) WHERE elementId(e) = $eid SET e.summary = $summary",
            eid=eid, summary=summary
        )

# ─── 메인 ───────────────────────────────────────────────
def main(dry_run=False, test_n=None):
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    
    limit = test_n if test_n else None
    entities = get_entities(driver, limit=limit)
    
    print(f"🔍 대상 엔티티: {len(entities)}개")
    print("─" * 50)

    results = []
    for i, ent in enumerate(entities, 1):
        name = ent["name"]
        labels = ent["labels"]
        eid = ent["eid"]
        label_str = ", ".join(l for l in labels if l != "Entity") or "일반"
        
        print(f"[{i}/{len(entities)}] {name} ({label_str})")
        
        if dry_run:
            print(f"  → [dry-run] 스킵\n")
            continue
        
        try:
            summary = query_llm(name, labels)
            update_summary(driver, eid, summary)
            print(f"  ✅ {summary[:80]}...\n" if len(summary) > 80 else f"  ✅ {summary}\n")
            results.append({"name": name, "labels": labels, "summary": summary})
        except Exception as e:
            print(f"  ❌ 오류: {e}\n")
            results.append({"name": name, "labels": labels, "error": str(e)})

    # 결과 저장
    if not dry_run and results:
        os.makedirs("temp_dev", exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = f"temp_dev/kg_enricher_{ts}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n📁 결과 저장: {out_path}")
        print(f"✅ 완료: {len([r for r in results if 'summary' in r])}/{len(entities)}개 성공")

    driver.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="실제 업데이트 없이 목록만 확인")
    parser.add_argument("--test", type=int, metavar="N", help="N개만 테스트 실행")
    args = parser.parse_args()
    
    main(dry_run=args.dry_run, test_n=args.test)
