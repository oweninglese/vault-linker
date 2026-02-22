from __future__ import annotations

import re
from typing import List

def normalize_md(md: str) -> str:
    md = md.replace("\r\n", "\n").replace("\r", "\n")
    lines = [ln.rstrip() for ln in md.split("\n")]
    md = "\n".join(lines)
    md = re.sub(r"\n{3,}", "\n\n", md)
    if not md.endswith("\n"):
        md += "\n"
    return md

def split_paragraphs(body: str) -> List[str]:
    body = body.replace("\r\n", "\n").replace("\r", "\n")
    parts = re.split(r"\n\s*\n", body.strip("\n"))
    return [p.strip("\n") for p in parts if p.strip() != ""]

def join_paragraphs(paragraphs: List[str]) -> str:
    return "\n\n".join(p.rstrip() for p in paragraphs).rstrip() + "\n"
