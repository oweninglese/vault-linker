
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import yaml


@dataclass(frozen=True)
class Note:
    path: Path
    text: str
    frontmatter: dict
    body: str


def _split_frontmatter(text: str) -> tuple[dict, str]:
    if not text.startswith("---"):
        return {}, text

    lines = text.splitlines(True)
    if not lines or lines[0].strip() != "---":
        return {}, text

    fm_lines = []
    i = 1
    while i < len(lines):
        if lines[i].strip() == "---":
            fm_text = "".join(fm_lines)
            body = "".join(lines[i + 1 :])
            try:
                data = yaml.safe_load(fm_text) or {}
                if not isinstance(data, dict):
                    data = {}
            except Exception:
                data = {}
            return data, body
        fm_lines.append(lines[i])
        i += 1

    return {}, text


def load_note(path: Path) -> Note:
    text = path.read_text(encoding="utf-8", errors="replace")
    fm, body = _split_frontmatter(text)
    return Note(path=path, text=text, frontmatter=fm, body=body)


def extract_tags(frontmatter: dict) -> list[str]:
    tags = frontmatter.get("tags")
    if tags is None:
        return []
    if isinstance(tags, str):
        s = tags.strip()
        return [s] if s else []
    if isinstance(tags, list):
        out = []
        for t in tags:
            if isinstance(t, str):
                s = t.strip()
                if s:
                    out.append(s)
        return out
    return []


_INLINE_TAG_RE = re.compile(r"(?<!\w)#([A-Za-z0-9][A-Za-z0-9_-]{1,63})")


def extract_inline_tags(body: str) -> set[str]:
    """
    Extracts simple inline tags like #Canada or #Treaty9.
    We intentionally do NOT try to parse multi-word tags here
    because Obsidian inline tags are typically single tokens.
    """
    if "#" not in body:
        return set()
    return set(m.group(1) for m in _INLINE_TAG_RE.finditer(body))
