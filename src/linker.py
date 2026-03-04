
from __future__ import annotations

import re


_CODE_FENCE = re.compile(r"```.*?```", re.DOTALL)
_WIKILINK = re.compile(r"\[\[.*?\]\]", re.DOTALL)


def _collect_skip_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for m in _CODE_FENCE.finditer(text):
        spans.append((m.start(), m.end()))
    for m in _WIKILINK.finditer(text):
        spans.append((m.start(), m.end()))
    spans.sort()
    return spans


def _in_spans(i: int, spans: list[tuple[int, int]]) -> bool:
    # spans sorted, linear scan with small count; can be binary searched if needed
    for a, b in spans:
        if i < a:
            return False
        if a <= i < b:
            return True
    return False


def link_first_per_file(body: str, matches: list[tuple[int, int, str]]) -> tuple[str, set[str], int]:
    """
    Insert [[Term]] for the first occurrence of each term in the file body.
    Avoid code fences and existing [[...]].
    Returns (new_body, found_terms, num_links_inserted).
    """
    if not matches:
        return body, set(), 0

    skip = _collect_skip_spans(body)
    linked: set[str] = set()
    found: set[str] = set()

    out_parts: list[str] = []
    cur = 0
    links_inserted = 0

    # matches are sorted by start already
    for start, end, term in matches:
        if start < cur:
            continue  # overlap / already consumed
        found.add(term)

        if term.lower() in linked:
            continue
        if _in_spans(start, skip):
            continue

        # write text before match
        out_parts.append(body[cur:start])
        out_parts.append(f"[[{term}]]")
        cur = end
        linked.add(term.lower())
        links_inserted += 1

    out_parts.append(body[cur:])
    new_body = "".join(out_parts)
    return new_body, found, links_inserted
