from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class Meta:
    title: Optional[str] = None
    author: Optional[str] = None
    source: Optional[str] = None
    created: Optional[str] = None  # year or ISO-like string

_RE_STABLE_URL = re.compile(r"(?im)^\s*Stable URL:\s*(\S+)\s*$")
_RE_SOURCE_URL = re.compile(r"(?im)^\s*(Source|URL):\s*(\S+)\s*$")
_RE_CHAPTER_TITLE = re.compile(r"(?im)^\s*Chapter Title:\s*(.+?)\s*$")
_RE_BOOK_TITLE = re.compile(r"(?im)^\s*Book Title:\s*(.+?)\s*$")
_RE_TITLE_LINE = re.compile(r"(?im)^\s*Title:\s*(.+?)\s*$")
_RE_AUTHOR = re.compile(r"(?im)^\s*(Author\(s\)|Chapter Author\(s\)|Author):\s*(.+?)\s*$")
_RE_PUBLISHED = re.compile(r"(?im)^\s*(Published|Publication Date):\s*(.+?)\s*$")
_RE_YEAR = re.compile(r"\b(18\d{2}|19\d{2}|20\d{2})\b")

def extract_metadata(title_fallback: str, body: str) -> Meta:
    """
    Uses only the first chunk to avoid scanning megabytes of PDFs.
    """
    head = "\n".join(body.splitlines()[:160])

    # Source URL preference: Stable URL > Source URL line
    source = None
    m = _RE_STABLE_URL.search(head)
    if m:
        source = m.group(1).strip()
    else:
        m2 = _RE_SOURCE_URL.search(head)
        if m2:
            source = m2.group(2).strip()

    # Title preference: Chapter Title > Title: > Book Title (fallback) > filename fallback
    title = None
    m = _RE_CHAPTER_TITLE.search(head)
    if m:
        title = m.group(1).strip()
    else:
        m = _RE_TITLE_LINE.search(head)
        if m:
            title = m.group(1).strip()
        else:
            m = _RE_BOOK_TITLE.search(head)
            if m:
                # Book title isn't the chapter/article title, but better than null
                title = m.group(1).strip()

    if not title:
        # As a last resort, use first non-empty line before boilerplate
        for ln in head.splitlines():
            ln = ln.strip()
            if not ln:
                continue
            if ln.lower().startswith(("jstor", "this content downloaded", "all use subject")):
                continue
            title = ln
            break

    if not title:
        title = title_fallback

    # Author
    author = None
    m = _RE_AUTHOR.search(head)
    if m:
        author = m.group(2).strip()

    # Created: year from Published line, else any year in header
    created = None
    m = _RE_PUBLISHED.search(head)
    if m:
        pub = m.group(2)
        ym = _RE_YEAR.search(pub)
        if ym:
            created = ym.group(1)
    if not created:
        ym = _RE_YEAR.search(head)
        if ym:
            created = ym.group(1)

    return Meta(title=title, author=author, source=source, created=created)
