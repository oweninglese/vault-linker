
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from datetime import datetime, timezone


_SCAN_MAX_LINES = 250

# Match "Key: value" lines (case-insensitive key)
_KEY_RE = re.compile(r"^\s*([A-Za-z0-9_() /-]{2,50})\s*:\s*(.*?)\s*$")

# URL patterns
_URL_RE = re.compile(r"https?://[^\s\])>\"']+")
_STABLE_URL_RE = re.compile(r"^\s*Stable\s+URL\s*:\s*(https?://\S+)\s*$", re.I)


def _norm_key(k: str) -> str:
    k = k.strip().lower()
    k = k.replace(" ", "_").replace("-", "_")
    return k


def mtime_rfc2822_utc(path: Path) -> str:
    """
    Format exactly like: Fri, 16 Jul 2021 02:20:30 UTC
    """
    ts = path.stat().st_mtime
    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    return dt.strftime("%a, %d %b %Y %H:%M:%S UTC")


def infer_title(frontmatter: dict, body: str, path: Path) -> str:
    # Priority: frontmatter title -> body "Title:" -> filename stem
    fm_title = frontmatter.get("title")
    if isinstance(fm_title, str) and fm_title.strip():
        return fm_title.strip()

    lines = body.splitlines()
    for raw in lines[:_SCAN_MAX_LINES]:
        m = _KEY_RE.match(raw)
        if not m:
            continue
        k = _norm_key(m.group(1))
        v = (m.group(2) or "").strip()
        if k == "title" and v:
            return v

    return path.stem


def extract_author(frontmatter: dict, body: str) -> str:
    # If already present in YAML, keep it unless empty
    for k in ("author", "authors", "source_author", "source_authors"):
        v = frontmatter.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()

    lines = body.splitlines()
    for raw in lines[:_SCAN_MAX_LINES]:
        m = _KEY_RE.match(raw)
        if not m:
            continue
        k_raw = m.group(1).strip()
        k = _norm_key(k_raw)
        v = (m.group(2) or "").strip()
        if not v:
            continue

        # Accept Author / Author(s) with uppercase A or any case
        if k in ("author", "authors", "author(s)", "author_s", "source_author", "source_authors"):
            return v

    return ""


def extract_source(frontmatter: dict, body: str) -> str:
    # If already present in YAML, keep it unless empty
    v = frontmatter.get("source")
    if isinstance(v, str) and v.strip():
        existing = v.strip()
    else:
        existing = ""

    # Priority 1: Stable URL:
    lines = body.splitlines()
    for raw in lines[:_SCAN_MAX_LINES]:
        m = _STABLE_URL_RE.match(raw)
        if m:
            return m.group(1).strip()

    # Priority 2: first URL in the first chunk (your “usually first link”)
    head = "\n".join(lines[:_SCAN_MAX_LINES])
    m2 = _URL_RE.search(head)
    if m2:
        return m2.group(0)

    # Fallback: keep existing
    return existing


def merge_tags(existing: object, found: set[str], min_len: int = 3) -> list[str]:
    """
    - Preserve existing order
    - Add new tags at end, sorted case-insensitively for stability
    - Dedup case-insensitively
    - Filter out too-short tags
    """
    out: list[str] = []
    seen: set[str] = set()

    def add_one(t: str) -> None:
        s = " ".join(t.strip().split())
        if not s or len(s) < min_len:
            return
        key = s.lower()
        if key not in seen:
            seen.add(key)
            out.append(s)

    if isinstance(existing, str):
        add_one(existing)
    elif isinstance(existing, list):
        for x in existing:
            if isinstance(x, str):
                add_one(x)

    for t in sorted(found, key=lambda x: x.lower()):
        if isinstance(t, str):
            add_one(t)

    return out


@dataclass(frozen=True)
class RequiredMeta:
    title: str
    author: str
    source: str
    created: str
