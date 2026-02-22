from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional
import re

def canonicalize_tag(s: str) -> str:
    """
    Canonical tag form used everywhere:
    - strip
    - collapse whitespace
    - convert spaces to hyphens
    - keep letters/numbers/hyphens/apostrophes
    - lowercase (stable)
    """
    s = (s or "").strip()
    if not s:
        return ""
    s = re.sub(r"\s+", " ", s)

    # remove surrounding quotes
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1].strip()

    # normalize curly apostrophe to straight
    s = s.replace("’", "'")

    # convert spaces/underscores to hyphen
    s = s.replace("_", " ").replace(" ", "-")

    # drop any characters we don't want in tags
    s = re.sub(r"[^A-Za-z0-9\-\']", "", s)

    # collapse repeated hyphens
    s = re.sub(r"-{2,}", "-", s).strip("-")

    return s.lower()

def normalize_tag_list(tags) -> List[str]:
    if tags is None:
        return []
    if isinstance(tags, str):
        tags = [tags]
    if not isinstance(tags, list):
        tags = [str(tags)]

    out: List[str] = []
    seen = set()
    for t in tags:
        c = canonicalize_tag(str(t))
        if not c:
            continue
        k = c.casefold()
        if k not in seen:
            out.append(c)
            seen.add(k)
    return out

def _split_registry_line(line: str) -> List[str]:
    """
    Accept common registry formats:
    - "#Canada"
    - "- Canada"
    - "* Canada"
    - "Canada, Aboriginal, Nunavut"
    - "Canada  # comment"
    """
    s = line.strip()
    if not s:
        return []
    if s.startswith("//") or s.startswith("# ") or s.startswith("##"):
        return []

    # remove inline comments after " #"
    s = re.sub(r"\s+#.*$", "", s).strip()

    # leading bullet markers
    s = re.sub(r"^[-*]\s+", "", s)

    # leading hashtag (tag notation)
    if s.startswith("#"):
        s = s[1:].strip()

    if not s:
        return []

    # comma-separated lists
    if "," in s:
        parts = [p.strip() for p in s.split(",")]
        return [p for p in parts if p]
    return [s]

def read_tags_file(path: Path) -> List[str]:
    path = Path(path).expanduser()
    if not path.exists():
        return []

    out: List[str] = []
    seen = set()
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        for part in _split_registry_line(raw):
            c = canonicalize_tag(part)
            if not c:
                continue
            k = c.casefold()
            if k not in seen:
                out.append(c)
                seen.add(k)
    return out
