"""Actionable prose errors, retaining the canonical house-style checks."""
import json
import re


_PASSAGE_MARKER = re.compile(r'\((P[0-9]+(?:[ ,;]+P[0-9]+)*)\)|\[(P[0-9]+(?:[ ,;]+P[0-9]+)*)\]')


def prose_errors(fields, *, passage_labels=()):
    from .stages import prose_problem
    errors = []
    displayed = set(passage_labels)
    def walk(node, parts):
        if isinstance(node, dict):
            for key, value in node.items():
                walk(value, [*parts, key])
        elif isinstance(node, list):
            for index, value in enumerate(node):
                walk(value, [*parts, index])
        elif isinstance(node, str):
            isolated = node
            for key in reversed(parts):
                isolated = [None]*key + [isolated] if isinstance(key,int) else {key:isolated}
            leaked = [match.group() for match in _PASSAGE_MARKER.finditer(node)
                      if displayed.intersection(re.findall(r'P[0-9]+', match.group()))]
            problem = ('Internal passage labels in public prose: ' + ', '.join(leaked)
                       + '; remove the markers from prose and retain passage references in evidence only'
                       if leaked else prose_problem(isolated))
            if problem:
                errors.append({'path':'/fields/'+'/'.join(str(p).replace('~','~0').replace('/','~1') for p in parts),
                    'rule':'prose', 'current':node[:400], 'message':problem,
                    'repair':'set this path to corrected text; retain supported facts and quoted titles'})
    walk(fields, [])
    return json.dumps(errors,ensure_ascii=False) if errors else ''
