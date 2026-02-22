from __future__ import annotations

from pathlib import Path
from typing import Iterable

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def iter_md_files(root: Path) -> Iterable[Path]:
    for p in sorted(root.rglob("*.md")):
        if p.is_file():
            yield p

def relpath_under(root: Path, p: Path) -> Path:
    return p.resolve().relative_to(root.resolve())

def read_text_utf8(p: Path) -> str:
    data = p.read_bytes()
    return data.decode("utf-8")

def write_text_utf8(p: Path, s: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(s, encoding="utf-8", newline="\n")
