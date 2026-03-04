
from __future__ import annotations

from pathlib import Path
import re
import yaml

from .meta import mtime_rfc2822_utc


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _has_frontmatter(text: str) -> bool:
    return text.startswith("---\n") or text.startswith("---\r\n")


def _strip_frontmatter(text: str) -> str:
    if not _has_frontmatter(text):
        return text
    lines = text.splitlines(True)
    if not lines or lines[0].strip() != "---":
        return text
    i = 1
    while i < len(lines):
        if lines[i].strip() == "---":
            return "".join(lines[i + 1 :]).lstrip("\n")
        i += 1
    return text


def _render_hub_frontmatter(title: str, created: str) -> str:
    fm = {"title": title, "created": created}
    y = yaml.safe_dump(fm, sort_keys=False, allow_unicode=True, width=88).strip()
    return "---\n" + y + "\n---\n\n"


def _upsert_block(text: str, start: str, end: str, new_block: str) -> str:
    pat = re.compile(re.escape(start) + r".*?" + re.escape(end), re.DOTALL)
    replacement = start + "\n" + new_block.rstrip() + "\n" + end
    m = pat.search(text)
    if m:
        return text[:m.start()] + replacement + text[m.end():]
    sep = "" if text.endswith("\n") or text == "" else "\n"
    return text + sep + replacement + "\n"


def render_hub_block(term: str, rel_paths: list[str]) -> str:
    lines = []
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


def normalize_hub_file(path: Path, marker_start: str, marker_end: str, title: str | None = None) -> bool:
    """
    If file contains hub markers, enforce:
      - frontmatter ONLY: title, created
    Returns True if file changed.
    """
    if not path.exists():
        return False
    old = path.read_text(encoding="utf-8", errors="replace")
    if marker_start not in old or marker_end not in old:
        return False

    body = _strip_frontmatter(old)
    # title: prefer provided; else use filename stem
    t = title if title else path.stem
    created = mtime_rfc2822_utc(path)
    new = _render_hub_frontmatter(t, created) + body.lstrip("\n")
    if new == old:
        return False
    path.write_text(new, encoding="utf-8")
    return True


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
    if path.exists():
        old = path.read_text(encoding="utf-8", errors="replace")

    body_wo_fm = _strip_frontmatter(old)

    # created is always the hub file's mtime AFTER write; we set it using current mtime for existing hubs.
    # For new hubs, we set created based on parent mtime; next run will correct it to the file's own mtime.
    created = mtime_rfc2822_utc(path) if path.exists() else mtime_rfc2822_utc(path.parent)

    fm = _render_hub_frontmatter(term, created)
    block = render_hub_block(term, rel_paths)
    normalized_body = _upsert_block(body_wo_fm, marker_start, marker_end, block)

    new = fm + normalized_body.lstrip("\n")

    if new == old:
        return False
    if not dry_run:
        path.write_text(new, encoding="utf-8")
    return True


def scrub_all_hubs(hub_root: Path, marker_start: str, marker_end: str) -> int:
    """
    Normalize every markdown file under hub_root that contains our marker block.
    This fixes 'mystery hubs' created by older runs or other scripts.
    """
    changed = 0
    for p in hub_root.rglob("*.md"):
        try:
            if normalize_hub_file(p, marker_start, marker_end):
                changed += 1
        except Exception:
            continue
    return changed
