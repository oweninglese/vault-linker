from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List
import re

from vault_linker.core.tags import normalize_tag_list

@dataclass(frozen=True)
class TagResult:
    tags: List[str]
    candidates: List[str]

def _uniq_ci(xs: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for x in xs:
        x = str(x).strip()
        if not x:
            continue
        k = x.casefold()
        if k not in seen:
            out.append(x)
            seen.add(k)
    return out

# Keep letters, digits, apostrophes, hyphens; everything else becomes space.
_KEEP = re.compile(r"[^a-z0-9\-\s'’]+", flags=re.IGNORECASE)

def _normalize_text_for_match(text: str) -> str:
    if not text:
        return ""
    t = text.replace("’", "'")
    t = _KEEP.sub(" ", t)
    t = t.replace("-", " ")  # hyphen ~ space equivalence
    t = re.sub(r"\s+", " ", t).strip().casefold()
    return t

def match_registry_tags(text: str, registry: List[str]) -> List[str]:
    """
    Robust matching:
    - casefold
    - punctuation tolerant
    - hyphen == space equivalence
    - single-word tags matched as tokens
    - multi-word tags matched as phrases with word boundaries
    """
    norm = _normalize_text_for_match(text)
    if not norm:
        return []

    padded = f" {norm} "
    tokens = set(norm.split())

    found: List[str] = []
    for tag in registry:
        if not tag:
            continue

        # registry tags are canonicalized elsewhere (usually lowercased + hyphenated),
        # but we normalize again for safety.
        phrase = _normalize_text_for_match(tag)
        if not phrase:
            continue

        if " " not in phrase:
            # token match
            if phrase in tokens:
                found.append(tag)
            continue

        # phrase match with boundaries
        if f" {phrase} " in padded:
            found.append(tag)

    return _uniq_ci(found)

def infer_tags(
    *,
    title: str,
    body: str,
    registry: List[str],
    yaml_tags,
    vocab_only: bool,
) -> TagResult:
    base = normalize_tag_list(yaml_tags)
    text = f"{title}\n\n{body}"

    if registry:
        matched = match_registry_tags(text, registry)
        tags_out = _uniq_ci(base + matched)
        return TagResult(tags=tags_out, candidates=[])

    # curated-first: no registry => no invented YAML tags
    return TagResult(tags=base, candidates=[])
