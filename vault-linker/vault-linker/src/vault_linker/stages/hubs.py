from __future__ import annotations

from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
import re

HubIndex = Dict[str, List[Tuple[str, str]]]

def build_hub_index() -> HubIndex:
    return defaultdict(list)

def _sanitize_display(s: str, fallback: str) -> str:
    s = (s or "").strip()
    if not s:
        s = fallback
    s = re.sub(r"^\s*#{1,6}\s+", "", s)
    s = re.sub(r"\s+", " ", s).strip()

    # reject giant CSV-like strings
    if s.count(",") >= 8 or len(s) > 180:
        s = fallback

    if len(s) > 140:
        s = s[:140].rstrip() + "…"
    return s

def add_refs(hub_index: HubIndex, rel_path: str, tags: List[str], title: str) -> None:
    fallback = Path(rel_path).stem
    display = _sanitize_display(title, fallback)
    for tag in tags:
        hub_index[tag].append((rel_path, display))

def write_hubs(output_root: Path, hub_index: HubIndex) -> None:
    hubs_dir = output_root / "_hubs"
    hubs_dir.mkdir(parents=True, exist_ok=True)

    for tag, refs in sorted(hub_index.items(), key=lambda kv: kv[0]):
        hub_path = hubs_dir / f"{tag}.md"

        # Deduplicate by rel_path (keep first display encountered)
        seen: Dict[str, str] = {}
        for rel_path, display in refs:
            if rel_path not in seen:
                seen[rel_path] = display

        lines: List[str] = []
        lines.append(f"# {tag}\n")
        lines.append("Referenced By:\n")

        for rel_path in sorted(seen.keys()):
            target = Path(rel_path).with_suffix("").as_posix()
            display = seen[rel_path]
            lines.append(f"- [[{target}|{display}]]")

        hub_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Optional: create an index note inside _hubs for browsing
    idx = hubs_dir / "_index.md"
    idx_lines = ["# Hubs\n", "Browse tag hubs:\n"]
    for tag in sorted(hub_index.keys()):
        idx_lines.append(f"- [[_hubs/{tag}|{tag}]]")
    idx.write_text("\n".join(idx_lines) + "\n", encoding="utf-8")
