from __future__ import annotations

import re
from typing import List, Tuple

# Reuse the same normalization idea as infer_tags for equivalence:
# - case-insensitive
# - hyphen/space equivalence when matching multiword tags
# But for LINKING we must replace the actual matched span in original text.

def _tag_to_regex(tag: str) -> re.Pattern:
    """
    Build a regex that matches the tag in text, case-insensitively, with:
    - hyphen and spaces equivalent separators
    - safe boundaries
    - preserves matched surface form
    """
    tag = (tag or "").strip()
    if not tag:
        return re.compile(r"(?!)")

    # Split tag into parts on hyphen/space
    parts = [p for p in re.split(r"[\s\-]+", tag) if p]
    if not parts:
        return re.compile(r"(?!)")

    # Separator in text can be spaces or hyphens (or multiple)
    sep = r"(?:[\s\-]+)"
    pat = sep.join(re.escape(p) for p in parts)

    # Boundaries: avoid matching inside words/ids
    return re.compile(rf"(?<![\w\-])({pat})(?![\w\-])", flags=re.IGNORECASE)

def _split_paragraphs(text: str) -> List[str]:
    # Keep blank lines as separators
    return re.split(r"(\n\s*\n)", text)

def _link_once_per_paragraph(par: str, tags: List[str]) -> str:
    """
    For each tag, replace the first match in this paragraph with a link.
    Keeps the matched surface text as the visible label:
      e.g., "CANADA" -> "[[canada|CANADA]]"
    """
    out = par
    for tag in tags:
        rx = _tag_to_regex(tag)

        def repl(m: re.Match) -> str:
            surface = m.group(1)
            # Keep surface text visible, link goes to canonical hub "tag"
            # If surface already equals tag (case-insensitive), still use alias form
            return f"[[{tag}|{surface}]]"

        out, n = rx.subn(repl, out, count=1)
    return out

def link_body(body: str, tags: List[str]) -> str:
    """
    Link first occurrence per paragraph for each tag.
    Paragraph = blocks separated by blank lines.
    """
    if not body or not tags:
        return body

    # Do not try to link absurdly large tag sets in one file (defensive)
    tags = [t for t in tags if t]
    if not tags:
        return body

    parts = _split_paragraphs(body)
    rebuilt: List[str] = []
    for part in parts:
        # separators like "\n\n" pass through
        if part.strip() == "":
            rebuilt.append(part)
            continue
        rebuilt.append(_link_once_per_paragraph(part, tags))
    return "".join(rebuilt)
