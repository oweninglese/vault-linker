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

# lines that look like YAML keys or boilerplate should never become titles
_BAD_TITLE_PREFIX = re.compile(
    r"(?i)^(author|created|source|tags|title)\s*:\s*"
)

def _sanitize_title(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return ""
    # Strip leading markdown heading markers
    s = re.sub(r"^\s*#{1,6}\s+", "", s)
    # collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    # cap length
    if len(s) > 140:
        s = s[:140].rstrip() + "…"
    return s

def _looks_like_csv_blob(s: str) -> bool:
    # crude but effective: lots of commas is usually a tags dump / CSV
    return s.count(",") >= 8

def _is_bad_fallback_title(s: str) -> bool:
    if not s:
        return True
    if _BAD_TITLE_PREFIX.match(s):
        return True
    if _looks_like_csv_blob(s):
        return True
    if len(s) > 200:
        return True
    return False

def extract_metadata(title_fallback: str, body: str) -> Meta:
    """
    Extract JSTOR-ish metadata; fallback is conservative and rejects garbage titles.
    """
    head = "\n".join((body or "").splitlines()[:220])

    # Source URL preference: Stable URL > Source URL line
    source = None
    m = _RE_STABLE_URL.search(head)
    if m:
        source = m.group(1).strip()
    else:
        m2 = _RE_SOURCE_URL.search(head)
        if m2:
            source = m2.group(2).strip()

    # Title preference: Chapter Title > Title: > Book Title (fallback) > safe first-line fallback > filename stem
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
                title = m.group(1).strip()

    title = _sanitize_title(title or "")

    if not title:
        # conservative fallback: first non-empty “real” line
        for ln in head.splitlines():
            ln = ln.strip()
            if not ln:
                continue
            low = ln.lower()
            if low.startswith(("jstor", "this content downloaded", "all use subject")):
                continue
            if _is_bad_fallback_title(ln):
                continue
            title = _sanitize_title(ln)
            break

    if not title or _is_bad_fallback_title(title):
        title = _sanitize_title(title_fallback) or title_fallback

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
