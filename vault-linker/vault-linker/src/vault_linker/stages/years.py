from __future__ import annotations

import re
from typing import List

# Match years 1800-2099 as standalone tokens (avoid IPs, long numbers)
_RE_YEAR = re.compile(r"(?<!\d)(18\d{2}|19\d{2}|20\d{2})(?!\d)")

def extract_year_tags(text: str, *, max_year_tags: int = 40) -> List[str]:
    years = _RE_YEAR.findall(text or "")
    # preserve order, de-dupe
    out: List[str] = []
    seen = set()
    for y in years:
        if y not in seen:
            out.append(y)
            seen.add(y)
        if len(out) >= max_year_tags:
            break
    return out
