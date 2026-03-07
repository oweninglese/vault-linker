from __future__ import annotations

import re


_CODE_FENCE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE = re.compile(r"`[^`\n]+`")
_WIKILINK = re.compile(r"\[\[.*?\]\]", re.DOTALL)
_MD_LINK = re.compile(r"\[[^\]]+\]\([^)]+\)")
_AUTOLINK = re.compile(r"<https?://[^>]+>")
_RAW_URL = re.compile(r"https?://[^\s)\]>]+")

# Paragraph separator: one or more blank lines, preserved during split/rejoin
_PARAGRAPH_SEP = re.compile(r"(\n\s*\n+)")


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


def _extract_existing_wikilink_targets(text: str) -> set[str]:
    """
    Return lowercased base targets already linked in [[...]] blocks.

    Examples:
      [[Canada]] -> canada
      [[Canada|CA]] -> canada
      [[Canada#History]] -> canada
      [[Canada#History|CA]] -> canada
    """
    linked: set[str] = set()

    for m in _WIKILINK.finditer(text):
        inner = m.group(0)[2:-2].strip()
        if not inner:
            continue

        target = inner.split("|", 1)[0].strip()
        if "#" in target:
            target = target.split("#", 1)[0].strip()

        if target:
            linked.add(target.lower())

    return linked


def _apply_mode_1_to_segment(segment: str, matches: list[tuple[int, int, str]]) -> tuple[str, set[str], int]:
    """
    First valid occurrence per segment.
    Used for file scope in mode 1 and paragraph scope in mode 2.
    """
    if not matches:
        return segment, set(), 0

    skip = _collect_skip_spans(segment)
    linked: set[str] = _extract_existing_wikilink_targets(segment)
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

        out_parts.append(segment[cur:start])
        out_parts.append(f"[[{term}]]")
        cur = end
        linked.add(key)
        links_inserted += 1

    out_parts.append(segment[cur:])
    return "".join(out_parts), found, links_inserted


def _apply_mode_3_to_segment(segment: str, matches: list[tuple[int, int, str]]) -> tuple[str, set[str], int]:
    """
    All valid occurrences in the segment.
    Existing wikilinks still block relinking because they are in skip spans.
    """
    if not matches:
        return segment, set(), 0

    skip = _collect_skip_spans(segment)
    found: set[str] = set()

    out_parts: list[str] = []
    cur = 0
    links_inserted = 0

    for start, end, term in matches:
        if start < cur:
            continue

        found.add(term)

        if _in_spans(start, skip):
            continue

        out_parts.append(segment[cur:start])
        out_parts.append(f"[[{term}]]")
        cur = end
        links_inserted += 1

    out_parts.append(segment[cur:])
    return "".join(out_parts), found, links_inserted


def _split_paragraphs_preserve(text: str) -> list[str]:
    """
    Split text into paragraph chunks and separator chunks, preserving separators.
    Example:
      ["para1", "\n\n", "para2", "\n\n\n", "para3"]
    """
    if not text:
        return [text]
    return _PARAGRAPH_SEP.split(text)


def _segment_matches(matches: list[tuple[int, int, str]], offset: int, seg_len: int) -> list[tuple[int, int, str]]:
    """
    Rebase absolute match offsets into segment-local offsets.
    """
    out: list[tuple[int, int, str]] = []
    seg_end = offset + seg_len
    for start, end, term in matches:
        if start >= offset and end <= seg_end:
            out.append((start - offset, end - offset, term))
    return out


def link_matches(
    body: str,
    matches: list[tuple[int, int, str]],
    linkify_mode: int = 1,
) -> tuple[str, set[str], int]:
    """
    linkify_mode:
      1 = first occurrence per file
      2 = first occurrence per paragraph
      3 = all valid occurrences
    """
    if linkify_mode not in (1, 2, 3):
        raise ValueError(f"Unsupported linkify_mode={linkify_mode}")

    if not matches:
        return body, set(), 0

    if linkify_mode == 1:
        return _apply_mode_1_to_segment(body, matches)

    if linkify_mode == 3:
        return _apply_mode_3_to_segment(body, matches)

    # mode 2: once per paragraph
    parts = _split_paragraphs_preserve(body)
    out_parts: list[str] = []
    found_all: set[str] = set()
    links_inserted = 0

    offset = 0
    for part in parts:
        local_matches = _segment_matches(matches, offset, len(part))

        # Paragraph separators should pass through unchanged.
        if _PARAGRAPH_SEP.fullmatch(part or ""):
            out_parts.append(part)
            offset += len(part)
            continue

        new_part, found, count = _apply_mode_1_to_segment(part, local_matches)
        out_parts.append(new_part)
        found_all.update(found)
        links_inserted += count
        offset += len(part)

    return "".join(out_parts), found_all, links_inserted


def link_first_per_file(body: str, matches: list[tuple[int, int, str]]) -> tuple[str, set[str], int]:
    return link_matches(body, matches, linkify_mode=1)
