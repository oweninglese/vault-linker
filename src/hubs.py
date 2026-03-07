from __future__ import annotations

from pathlib import Path
import re
import yaml

from .meta import mtime_rfc2822_utc


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _has_frontmatter(text: str) -> bool:
    return text.startswith("---\n") or text.startswith("---\r\n")


def _split_frontmatter(text: str) -> tuple[dict, str]:
    if not _has_frontmatter(text):
        return {}, text

    lines = text.splitlines(True)
    if not lines or lines[0].strip() != "---":
        return {}, text

    fm_lines: list[str] = []
    i = 1
    while i < len(lines):
        if lines[i].strip() == "---":
            fm_text = "".join(fm_lines)
            body = "".join(lines[i + 1 :]).lstrip("\n")
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


def _render_hub_frontmatter(title: str, created: str) -> str:
    fm = {"title": title, "created": created}
    y = yaml.safe_dump(fm, sort_keys=False, allow_unicode=True, width=88).strip()
    return "---\n" + y + "\n---\n\n"


def is_managed_hub(text: str, marker_start: str, marker_end: str) -> bool:
    return marker_start in text and marker_end in text


def _upsert_block(text: str, start: str, end: str, new_block: str) -> str:
    pat = re.compile(re.escape(start) + r".*?" + re.escape(end), re.DOTALL)
    replacement = start + "\n" + new_block.rstrip() + "\n" + end
    m = pat.search(text)
    if m:
        return text[:m.start()] + replacement + text[m.end():]
    sep = "" if text.endswith("\n") or text == "" else "\n"
    return text + sep + replacement + "\n"


def render_hub_block(term: str, rel_paths: list[str]) -> str:
    lines: list[str] = []
    lines.append(f"## Backlinks to {term}")
    lines.append("")
    lines.append(f"_Mentions: {len(rel_paths)}_")
    lines.append("")
    if not rel_paths:
        lines.append("_No notes mention this yet._")
        return "\n".join(lines)

    for rel in rel_paths:
        target = rel[:-3] if rel.lower().endswith(".md") else rel
        lines.append(f"- [[{target}]]")
    return "\n".join(lines)


def update_hub_page(
    path: Path,
    marker_start: str,
    marker_end: str,
    term: str,
    rel_paths: list[str],
    dry_run: bool,
) -> bool:
    _ensure_parent(path)

    old = ""
    old_fm: dict = {}
    old_body = ""

    if path.exists():
        old = path.read_text(encoding="utf-8", errors="replace")
        old_fm, old_body = _split_frontmatter(old)

    created = ""
    if isinstance(old_fm.get("created"), str) and old_fm.get("created").strip():
        created = old_fm["created"].strip()
    else:
        created = mtime_rfc2822_utc(path) if path.exists() else mtime_rfc2822_utc(path.parent)

    title = ""
    if isinstance(old_fm.get("title"), str) and old_fm.get("title").strip():
        title = old_fm["title"].strip()
    else:
        title = term

    body_source = old_body if path.exists() else ""
    block = render_hub_block(term, rel_paths)
    normalized_body = _upsert_block(body_source, marker_start, marker_end, block)

    new = _render_hub_frontmatter(title, created) + normalized_body.lstrip("\n")

    if new == old:
        return False

    if not dry_run:
        path.write_text(new, encoding="utf-8")

    return True
