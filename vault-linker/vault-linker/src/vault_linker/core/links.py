from __future__ import annotations

import re
from typing import Dict, List, Tuple

def _is_already_linked(paragraph: str, start: int, end: int) -> bool:
    # wiki link guard
    window = paragraph[max(0, start - 2) : min(len(paragraph), end + 2)]
    if "[[" in window and "]]" in window:
        return True
    # markdown link guard (crude but safe)
    if start > 0 and paragraph[start - 1] in "[(":
        return True
    if end < len(paragraph) and paragraph[end : end + 1] in "]":
        return True
    return False

def link_once_per_paragraph(
    paragraph: str,
    targets: List[str],
    *,
    wiki: bool = True,
) -> Tuple[str, Dict[str, int]]:
    counts: Dict[str, int] = {}
    out = paragraph

    for t in targets:
        pat = re.compile(rf"(?<![\w\-]){re.escape(t)}(?![\w\-])")
        m = pat.search(out)
        if not m:
            continue
        if _is_already_linked(out, m.start(), m.end()):
            continue
        repl = f"[[{t}]]" if wiki else f"[{t}]({t}.md)"
        out = out[: m.start()] + repl + out[m.end() :]
        counts[t] = 1
    return out, counts
