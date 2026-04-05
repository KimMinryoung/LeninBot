---
name: kg-maintenance
description: Maintains Neo4j Knowledge Graph quality: merges duplicate entities, assigns missing entity types (Person/Organization/Location/Event/Concept), removes orphaned nodes, and validates relationship integrity. Use when KG has duplicate entities, untyped nodes, or after bulk data ingestion. Keywords: KG 정리, 중복 병합, 엔티티 타입, 고아 노드, knowledge graph cleanup, deduplication.
compatibility: Requires Neo4j connection and Python 3.10+.
metadata:
  author: cyber-lenin
  version: "2.0"
allowed-tools: kg_admin read_self execute_python
---

# KG Maintenance Skill

## When to run
- After bulk data ingestion
- When `read_self(source="kg_status")` shows 100+ untyped entities
- When duplicate entities are suspected (e.g. "미국" vs "USA" vs "United States")
- Weekly cleanup (every Monday)

## Step 1 — Check current status
```
read_self(source="kg_status")
```
Review total entities, edges, recent episodes.

## Step 2 — Find duplicate entities
```
kg_admin(action="query", query="MATCH (a), (b) WHERE a.name CONTAINS b.name AND id(a) <> id(b) RETURN a.name, b.name LIMIT 50")
```
Evaluate candidates manually:
- Same entity, different spelling → merge (prefer Korean name)
- Similar but distinct entities → do NOT merge

## Step 3 — Merge duplicates
```
kg_admin(action="merge_entities", source_name="USA", target_name="미국")
```
This merges the source entity INTO the target, transferring all relationships.

## Step 4 — Assign missing types
Run the existing classification script:
```python
import subprocess, os
result = subprocess.run(
    [os.environ.get("VENV_PYTHON", "python"), "scripts/classify_untyped_entities.py"],
    capture_output=True, text=True,
    cwd=os.environ.get("PROJECT_ROOT", "/home/grass/leninbot"),
    timeout=120,
)
print(result.stdout[-2000:])
```
Assigns `Person/Organization/Location/Event/Concept` types via LLM inference, batch of 50.

## Step 5 — Find orphan nodes
```
kg_admin(action="query", query="MATCH (n) WHERE NOT (n)--() RETURN n.name, n.type LIMIT 50")
```
Review orphans. If truly disconnected and not in any episode, delete the episode:
```
kg_admin(action="delete_episode", episode_name="episode_name_here")
```

## Step 6 — Validate
```
kg_admin(action="query", query="MATCH (n) RETURN n.type, count(n) ORDER BY count(n) DESC")
```
Check type distribution for anomalies.

## Cautions
- Merges are irreversible. Skip if uncertain.
- Delete only after confirming the node is truly orphaned.
