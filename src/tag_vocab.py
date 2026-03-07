from __future__ import annotations

from pathlib import Path


def load_terms_from_tagfile(path: Path) -> list[str]:
    """
    Accepts:
      - comma-separated values
      - one term per line
      - optional comment lines starting with // or #

    Returns normalized terms, deduped in stable order.
    """
    text = path.read_text(encoding="utf-8-sig", errors="strict")

    seen: set[str] = set()
    out: list[str] = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("//") or line.startswith("#"):
            continue
        if "//" in line:
            line = line.split("//", 1)[0].strip()
            if not line:
                continue

        for part in line.split(","):
            t = " ".join(part.strip().split())
            if not t:
                continue
            key = t.lower()
            if key not in seen:
                seen.add(key)
                out.append(t)

    return out
