from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from vault_linker.io.fs import write_text_utf8

@dataclass
class HubIndex:
    refs: Dict[str, List[str]]

def build_hub_index() -> HubIndex:
    return HubIndex(refs={})

def add_refs(index: HubIndex, rel_doc: str, tags: List[str]) -> None:
    for t in tags:
        index.refs.setdefault(t, []).append(rel_doc)

def write_hubs(output_root: Path, index: HubIndex) -> None:
    for tag in sorted(index.refs.keys(), key=lambda x: x.casefold()):
        refs = sorted(set(index.refs[tag]), key=lambda x: x.casefold())
        lines = [f"# {tag}", "", "Referenced By:", ""]
        for r in refs:
            lines.append(f"- {r}")
        lines.append("")
        write_text_utf8(output_root / f"{tag}.md", "\n".join(lines))
