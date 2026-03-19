#!/usr/bin/env python3
"""
KG 중복 엔티티 탐지 스크립트
유사도 기반으로 병합 후보 쌍을 추출한다.
"""
import os
import sys
from neo4j import GraphDatabase
from difflib import SequenceMatcher

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASSWORD", "")
THRESHOLD = 0.85  # 유사도 임계값

def similarity(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def find_duplicates():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
    with driver.session() as session:
        result = session.run("MATCH (n:Entity) RETURN n.name AS name ORDER BY name")
        names = [r["name"] for r in result if r["name"]]

    candidates = []
    for i, a in enumerate(names):
        for b in names[i+1:]:
            score = similarity(a, b)
            if score >= THRESHOLD and a != b:
                candidates.append((a, b, round(score, 3)))

    candidates.sort(key=lambda x: -x[2])
    driver.close()
    return candidates

if __name__ == "__main__":
    dupes = find_duplicates()
    print(f"중복 후보 {len(dupes)}쌍 발견:\n")
    for src, tgt, score in dupes[:50]:
        print(f"  [{score}] '{src}' ↔ '{tgt}'")
    if not dupes:
        print("중복 없음.")
