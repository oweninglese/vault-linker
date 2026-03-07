from __future__ import annotations

import re


_CODE_FENCE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE = re.compile(r"`[^`\n]+`")
_WIKILINK = re.compile(r"\[\[.*?\]\]", re.DOTALL)
_MD_LINK = re.compile(r"\[[^\]]+\]\([^)]+\)")
_AUTOLINK = re.compile(r"<https?://[^>]+>")
_RAW_URL = re.compile(r"https?://[^\s)\]>]+")


def _collect_skip_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []

    for rx in (_CODE_FENCE, _INLINE_CODE, _WIKILINK, _MD_LINK, _AUTOLINK, _RAW_URL):
        for m in rx.finditer(text):
            spans.append((m.start(), m.end()))

    spans.sort()
    return spans


def _in_spans(i: int, spans: list[tuple[int, int]]) -> bool:
    for a, b in spans:
        if i < a:
            return False
        if a <= i < b:
            return True
    return False


def link_first_per_file(body: str, matches: list[tuple[int, int, str]]) -> tuple[str, set[str], int]:
    if not matches:
        return body, set(), 0

    skip = _collect_skip_spans(body)
    linked: set[str] = set()
    found: set[str] = set()

    out_parts: list[str] = []
    cur = 0
    links_inserted = 0

    for start, end, term in matches:
        if start < cur:
            continue

        found.add(term)

        key = term.lower()
        if key in linked:
            continue
        if _in_spans(start, skip):
            continue

        out_parts.append(body[cur:start])
        out_parts.append(f"[[{term}]]")
        cur = end
        linked.add(key)
        links_inserted += 1

    out_parts.append(body[cur:])
    return "".join(out_parts), found, links_inserted
