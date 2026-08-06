#!/usr/bin/env python3
"""
타입 없는 KG 엔티티에 Person/Organization/Location/Event/Concept 부여.
배치 50개씩 처리. LLM 호출은 게이트웨이 경유(llm.call_registry).
"""
import os
import json
from neo4j import GraphDatabase

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASSWORD", "")
BATCH_SIZE = 50
VALID_TYPES = {"Person", "Organization", "Location", "Event", "Concept"}

def get_untyped(session):
    result = session.run(
        "MATCH (n:Entity) WHERE n.type IS NULL OR n.type = '' RETURN n.name AS name LIMIT 500"
    )
    return [r["name"] for r in result if r["name"]]

def classify_batch(names):
    prompt = (
        "다음 엔티티 목록을 Person/Organization/Location/Event/Concept 중 하나로 분류하라.\n"
        "JSON 배열로만 응답. 예: [{\"name\": \"김정은\", \"type\": \"Person\"}, ...]\n\n"
        f"엔티티 목록:\n{json.dumps(names, ensure_ascii=False)}"
    )
    from llm.call_registry import generate_sync

    text = (generate_sync("kg_type_assignment", prompt) or "").strip()
    if not text:
        raise RuntimeError(
            "kg_type_assignment: 게이트웨이가 본문을 돌려주지 않았다 "
            "(원인은 llm_gateway.audit / [llm-registry] 경고 참조)"
        )
    # JSON 추출
    start = text.find("[")
    end = text.rfind("]") + 1
    return json.loads(text[start:end])

def assign_types():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
    updated = 0

    with driver.session() as session:
        untyped = get_untyped(session)
        print(f"타입 없는 엔티티: {len(untyped)}개")

        for i in range(0, len(untyped), BATCH_SIZE):
            batch = untyped[i:i+BATCH_SIZE]
            print(f"배치 {i//BATCH_SIZE + 1} 처리 중... ({len(batch)}개)")
            try:
                classified = classify_batch(batch)
                for item in classified:
                    name = item.get("name")
                    etype = item.get("type")
                    if name and etype in VALID_TYPES:
                        session.run(
                            "MATCH (n:Entity {name: $name}) SET n.type = $type",
                            name=name, type=etype
                        )
                        updated += 1
            except Exception as e:
                print(f"  배치 오류: {e}")
                continue

    driver.close()
    print(f"\n완료: {updated}개 엔티티 타입 부여됨")

if __name__ == "__main__":
    assign_types()
