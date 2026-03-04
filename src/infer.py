
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

# --- Guardrails / stoplists (small but effective) ---
STOPWORDS = {
    "the","a","an","and","or","but","if","then","else","so","to","of","in","on","at","by","for","with","from",
    "as","is","are","was","were","be","been","being","this","that","these","those","it","its","they","them","their",
    "we","our","you","your","i","me","my","he","him","his","she","her","hers",
}
MONTHS = {"january","february","march","april","may","june","july","august","september","october","november","december"}
DAYS = {"monday","tuesday","wednesday","thursday","friday","saturday","sunday"}

# Tokens like "private-equity-ish" or "randomized-control-trial"
HYPHEN_TOKEN = re.compile(r"\b[a-z]+(?:-[a-z]+){1,6}\b", re.I)

# Acronyms like WHO, UN, NATO
ACRONYM = re.compile(r"\b[A-Z]{2,6}\b")

# Title Case words (allow internal apostrophes)
TC_WORD = r"[A-Z][a-z]+(?:'[A-Za-z]+)?"
TITLECASE_PHRASE = re.compile(rf"\b{TC_WORD}(?:\s+{TC_WORD}){{1,4}}\b")  # 2..5 words

# Mid-sentence proper noun-ish: single TitleCase tokens, not at sentence start.
# We approximate sentence starts by punctuation + whitespace.
SENT_START = re.compile(r"(^|[.!?]\s+)([A-Z][a-z]+)\b")

# “Capitalized words mid-sentence” capture (single tokens)
CAP_TOKEN = re.compile(r"\b[A-Z][a-z]{2,}\b")

# Basic URL/code noise removal
CODE_FENCE = re.compile(r"```.*?```", re.DOTALL)
WIKILINK = re.compile(r"\[\[.*?\]\]", re.DOTALL)
URL = re.compile(r"https?://\S+")


@dataclass(frozen=True)
class InferConfig:
    min_len: int = 3
    max_chars: int = 50
    max_words: int = 5
    include_acronyms: bool = False


def _normalize(s: str) -> str:
    s = " ".join(s.strip().split())
    # Normalize weird quotes
    s = s.replace("“", '"').replace("”", '"').replace("’", "'")
    return s


def _is_bad_candidate(s: str, cfg: InferConfig) -> bool:
    if not s:
        return True
    if len(s) < cfg.min_len:
        return True
    if len(s) > cfg.max_chars:
        return True
    if s.isdigit():
        return True
    words = s.split()
    if len(words) > cfg.max_words:
        return True
    low = s.lower()
    if low in STOPWORDS or low in MONTHS or low in DAYS:
        return True
    # Reject if mostly punctuation
    if sum(ch.isalnum() for ch in s) < max(2, len(s) // 3):
        return True
    return False


def infer_candidates(title: str, body: str, cfg: InferConfig) -> set[str]:
    """
    Extract candidate tag terms from title + body.
    Does NOT return approved tags; it returns suggestions.
    """
    title = _normalize(title or "")
    body = body or ""

    # Remove big noise zones
    cleaned = CODE_FENCE.sub(" ", body)
    cleaned = WIKILINK.sub(" ", cleaned)
    cleaned = URL.sub(" ", cleaned)

    cands: set[str] = set()

    # --- Title-driven inference ---
    for m in TITLECASE_PHRASE.finditer(title):
        s = _normalize(m.group(0))
        if not _is_bad_candidate(s, cfg):
            cands.add(s)

    for m in HYPHEN_TOKEN.finditer(title):
        s = _normalize(m.group(0))
        if not _is_bad_candidate(s, cfg):
            cands.add(s)

    # --- Body-driven inference ---
    # 1) Multi-word TitleCase phrases anywhere
    for m in TITLECASE_PHRASE.finditer(cleaned):
        s = _normalize(m.group(0))
        if not _is_bad_candidate(s, cfg):
            cands.add(s)

    # 2) Hyphenated tokens (often good domain terms)
    for m in HYPHEN_TOKEN.finditer(cleaned):
        s = _normalize(m.group(0))
        if not _is_bad_candidate(s, cfg):
            cands.add(s)

    # 3) Mid-sentence single capitalized tokens (more risky; filter hard)
    # We only keep if the token is not a month/day/stopword and not at sentence start.
    # We approximate sentence starts by checking positions matched by SENT_START.
    sentence_starts = {m.start(2) for m in SENT_START.finditer(cleaned)}
    for m in CAP_TOKEN.finditer(cleaned):
        if m.start() in sentence_starts:
            continue
        s = _normalize(m.group(0))
        if not _is_bad_candidate(s, cfg):
            cands.add(s)

    # 4) Acronyms (optional)
    if cfg.include_acronyms:
        for m in ACRONYM.finditer(cleaned):
            s = m.group(0)
            if not _is_bad_candidate(s, cfg):
                cands.add(s)

    return cands


def aggregate_candidates(items: list[tuple[str, set[str]]]) -> tuple[Counter, dict[str, list[str]]]:
    """
    items: list of (rel_path, candidates)
    returns:
      - Counter(term -> count)
      - examples: term -> up to 5 example rel_paths
    """
    counts: Counter = Counter()
    examples: dict[str, list[str]] = {}
    for rel, cset in items:
        for term in cset:
            counts[term] += 1
            ex = examples.setdefault(term, [])
            if len(ex) < 5:
                ex.append(rel)
    return counts, examples
