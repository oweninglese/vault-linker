
from __future__ import annotations

from pathlib import Path


def load_terms_from_tagfile(path: Path) -> list[str]:
    """
    Tag vocab is a plain text file containing comma-separated values.
    Also tolerates one term per line.
    Example:
      Canada, Treaty 9, COVID-19
      1934, 1935

    Returns a list of normalized terms (deduped, stable order).
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    seen: set[str] = set()
    out: list[str] = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("//"):
            continue
        if "//" in line:
            line = line.split("//", 1)[0].strip()
            if not line:
                continue

        for part in line.split(","):
            t = part.strip()
            if not t:
                continue
            if t.startswith("#"):
                t = t[1:].strip()
            if t.startswith("-"):
                t = t[1:].strip()
            t = " ".join(t.split())
            if t and t.lower() not in seen:
                seen.add(t.lower())
                out.append(t)

    return out
