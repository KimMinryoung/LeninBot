"""Lossless source references; IDs select citations, never infer evidence."""
from copy import deepcopy
import re


def resolve_evidence_sources(evidence, citations):
    if not isinstance(citations, list) or any(not isinstance(c, str) or not c.strip() for c in citations):
        raise ValueError("citations must be a list of non-empty source references")
    result = deepcopy(evidence)
    if not isinstance(result, list):
        return result  # the ordinary schema validator explains the type error
    for index, item in enumerate(result):
        if not isinstance(item, dict) or "source_id" not in item:
            continue
        source_id = item.pop("source_id")
        match = re.fullmatch(r"S([1-9][0-9]*)", str(source_id))
        position = int(match[1]) - 1 if match else -1
        if not 0 <= position < len(citations):
            raise ValueError(f"evidence[{index}].source_id must select S1..S{len(citations)} from citations in order")
        source = citations[position].strip()
        if "source" in item and item["source"] != source:
            raise ValueError(f"evidence[{index}].source and source_id disagree")
        item["source"] = source
    return result
