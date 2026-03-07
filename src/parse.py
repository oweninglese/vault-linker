from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
import yaml

from .diagnostics import Diagnostic, Severity


@dataclass(frozen=True)
class Note:
    path: Path
    text: str
    encoding: str
    has_frontmatter: bool
    frontmatter_valid: bool
    frontmatter: dict
    raw_frontmatter: str | None
    body: str
    diagnostics: list[Diagnostic] = field(default_factory=list)


_INLINE_TAG_RE = re.compile(r"(?<!\w)#([A-Za-z0-9][A-Za-z0-9_-]{1,63})")


def _split_frontmatter_lexically(text: str) -> tuple[bool, str | None, str]:
    if not text.startswith("---"):
        return False, None, text

    lines = text.splitlines(True)
    if not lines or lines[0].strip() != "---":
        return False, None, text

    fm_lines: list[str] = []
    i = 1
    while i < len(lines):
        if lines[i].strip() == "---":
            raw_fm = "".join(fm_lines)
            body = "".join(lines[i + 1 :])
            return True, raw_fm, body
        fm_lines.append(lines[i])
        i += 1

    return False, None, text


def parse_note_text(path: Path, text: str, encoding: str) -> Note:
    diags: list[Diagnostic] = []

    has_fm, raw_fm, body = _split_frontmatter_lexically(text)
    if not has_fm:
        return Note(
            path=path,
            text=text,
            encoding=encoding,
            has_frontmatter=False,
            frontmatter_valid=True,
            frontmatter={},
            raw_frontmatter=None,
            body=text,
            diagnostics=diags,
        )

    frontmatter: dict = {}
    valid = True
    try:
        loaded = yaml.safe_load(raw_fm or "")
        if loaded is None:
            loaded = {}
        if not isinstance(loaded, dict):
            valid = False
            diags.append(
                Diagnostic(
                    code="YAML_PARSE_FAIL",
                    message="Frontmatter exists but did not parse to a mapping.",
                    severity=Severity.WARNING,
                    path=str(path),
                )
            )
        else:
            frontmatter = loaded
    except Exception as e:
        valid = False
        diags.append(
            Diagnostic(
                code="YAML_PARSE_FAIL",
                message="Frontmatter YAML could not be parsed.",
                severity=Severity.WARNING,
                path=str(path),
                context=repr(e),
            )
        )

    return Note(
        path=path,
        text=text,
        encoding=encoding,
        has_frontmatter=True,
        frontmatter_valid=valid,
        frontmatter=frontmatter if valid else {},
        raw_frontmatter=raw_fm,
        body=body,
        diagnostics=diags,
    )


def render_frontmatter(frontmatter: dict) -> str:
    preferred = ["title", "author", "source", "created", "tags"]
    out: dict = {}

    for key in preferred:
        if key in frontmatter:
            out[key] = frontmatter[key]
    for key, value in frontmatter.items():
        if key not in out:
            out[key] = value

    y = yaml.safe_dump(
        out,
        sort_keys=False,
        allow_unicode=True,
        width=88,
        default_flow_style=False,
    ).strip()

    return f"---\n{y}\n---\n\n"


def extract_tags(frontmatter: dict) -> list[str]:
    tags = frontmatter.get("tags")
    if tags is None:
        return []
    if isinstance(tags, str):
        s = tags.strip()
        return [s] if s else []
    if isinstance(tags, list):
        out: list[str] = []
        for t in tags:
            if isinstance(t, str):
                s = t.strip()
                if s:
                    out.append(s)
        return out
    return []


def extract_inline_tags(body: str) -> set[str]:
    if "#" not in body:
        return set()
    return {m.group(1) for m in _INLINE_TAG_RE.finditer(body)}
