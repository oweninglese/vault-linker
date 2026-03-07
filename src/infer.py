from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass


STOPWORDS = {
    "the","a","an","and","or","but","if","then","else","so","to","of","in","on","at","by","for","with","from",
    "as","is","are","was","were","be","been","being","this","that","these","those","it","its","they","them","their",
    "we","our","you","your","i","me","my","he","him","his","she","her","hers",
}

MONTHS = {
    "january","february","march","april","may","june","july","august",
    "september","october","november","december",
}

DAYS = {
    "monday","tuesday","wednesday","thursday","friday","saturday","sunday",
}

NUMBER_WORDS = {
    "zero","one","two","three","four","five","six","seven","eight","nine","ten",
    "eleven","twelve","thirteen","fourteen","fifteen","sixteen","seventeen","eighteen","nineteen","twenty",
    "first","second","third","fourth","fifth","sixth","seventh","eighth","ninth","tenth",
}

GENERIC_SINGLE_WORDS = {
    "history","region","north","south","east","west","northern","southern","eastern","western",
    "treaty","agreement","chapter","section","article","page","pages","part","volume","report",
    "study","paper","project","program","system","model","framework","theory",
}

ALLOWED_SINGLE_WORDS = {
    "Canada","Ontario","Quebec","Moose","Cree","James","Bay",
}

CODE_FENCE = re.compile(r"```.*?```", re.DOTALL)
INLINE_CODE = re.compile(r"`[^`\n]+`")
WIKILINK = re.compile(r"\[\[.*?\]\]", re.DOTALL)
URL = re.compile(r"https?://\S+")

TC_WORD = r"[A-Z][a-z]+(?:'[A-Za-z]+)?"
TITLECASE_PHRASE = re.compile(rf"\b{TC_WORD}(?:\s+{TC_WORD}){{1,4}}\b")
CAP_TOKEN = re.compile(r"\b[A-Z][a-z]{2,}\b")
ACRONYM = re.compile(r"\b[A-Z]{2,6}\b")
HYPHEN_TOKEN = re.compile(r"\b[A-Za-z]+(?:-[A-Za-z0-9]+){1,6}\b")

# Keep: capitalized phrase + number, e.g. "Treaty 9", "Article 23", "Phase 2"
CAP_NUM_PHRASE = re.compile(rf"\b{TC_WORD}(?:\s+{TC_WORD}){{0,3}}\s+\d{{1,4}}[A-Za-z]?\b")

# Keep: acronym/uppercase + number, e.g. "G7", "B52", "COVID-19"
ACRONYM_NUM = re.compile(r"\b(?:[A-Z]{1,8}(?:-\d{1,4}[A-Za-z]?|\d{1,4}[A-Za-z]?))\b")

# Intentionally removed:
# broad lowercase word+number matching such as "and 300", "almost 40"


@dataclass(frozen=True)
class InferConfig:
    min_len: int = 3
    max_chars: int = 60
    max_words: int = 6
    include_acronyms: bool = False


def _normalize(s: str) -> str:
    s = " ".join((s or "").strip().split())
    s = s.replace("“", '"').replace("”", '"').replace("’", "'")
    return s


def _is_bad_candidate(s: str, cfg: InferConfig) -> bool:
    if not s:
        return True

    s = _normalize(s)
    if len(s) < cfg.min_len:
        return True
    if len(s) > cfg.max_chars:
        return True

    words = s.split()
    if len(words) > cfg.max_words:
        return True

    low = s.lower()

    if low in STOPWORDS or low in MONTHS or low in DAYS or low in NUMBER_WORDS:
        return True

    if s.isdigit():
        return True

    if sum(ch.isalnum() for ch in s) < max(2, len(s) // 3):
        return True

    if len(words) == 1:
        if low in GENERIC_SINGLE_WORDS:
            return True
        if low in NUMBER_WORDS:
            return True

        # Allow acronym+number style tokens
        if re.fullmatch(r"(?:[A-Z]{1,8}(?:-\d{1,4}[A-Za-z]?|\d{1,4}[A-Za-z]?))", s):
            return False

        if s not in ALLOWED_SINGLE_WORDS:
            if not re.fullmatch(r"[A-Z][a-z]{3,}", s):
                return True

    return False


def _clean_text(body: str) -> str:
    cleaned = CODE_FENCE.sub(" ", body or "")
    cleaned = INLINE_CODE.sub(" ", cleaned)
    cleaned = WIKILINK.sub(" ", cleaned)
    cleaned = URL.sub(" ", cleaned)
    return cleaned


def _suppress_shorter_overlaps(cands: set[str]) -> set[str]:
    ordered = sorted(cands, key=lambda s: (-len(s.split()), -len(s), s.lower()))
    kept: list[str] = []

    for cand in ordered:
        cand_low = cand.lower()
        is_subsumed = False

        for longer in kept:
            longer_low = longer.lower()
            if cand_low == longer_low:
                is_subsumed = True
                break
            if re.search(rf"(?<!\w){re.escape(cand_low)}(?!\w)", longer_low):
                is_subsumed = True
                break

        if not is_subsumed:
            kept.append(cand)

    return set(kept)


def infer_candidates(title: str, body: str, cfg: InferConfig) -> set[str]:
    title = _normalize(title or "")
    cleaned = _clean_text(body or "")

    cands: set[str] = set()

    # Title
    for rx in (CAP_NUM_PHRASE, TITLECASE_PHRASE, ACRONYM_NUM, HYPHEN_TOKEN):
        for m in rx.finditer(title):
            s = _normalize(m.group(0))
            if not _is_bad_candidate(s, cfg):
                cands.add(s)

    # Body
    for m in CAP_NUM_PHRASE.finditer(cleaned):
        s = _normalize(m.group(0))
        if not _is_bad_candidate(s, cfg):
            cands.add(s)

    for m in TITLECASE_PHRASE.finditer(cleaned):
        s = _normalize(m.group(0))
        if not _is_bad_candidate(s, cfg):
            cands.add(s)

    for rx in (ACRONYM_NUM, HYPHEN_TOKEN):
        for m in rx.finditer(cleaned):
            s = _normalize(m.group(0))
            if not _is_bad_candidate(s, cfg):
                cands.add(s)

    for m in CAP_TOKEN.finditer(cleaned):
        s = _normalize(m.group(0))
        if not _is_bad_candidate(s, cfg):
            cands.add(s)

    if cfg.include_acronyms:
        for m in ACRONYM.finditer(cleaned):
            s = m.group(0)
            if not _is_bad_candidate(s, cfg):
                cands.add(s)

    cands = _suppress_shorter_overlaps(cands)
    return cands


def aggregate_candidates(items: list[tuple[str, set[str]]]) -> tuple[Counter, dict[str, list[str]]]:
    counts: Counter = Counter()
    examples: dict[str, list[str]] = {}

    for rel, cset in items:
        for term in cset:
            counts[term] += 1
            ex = examples.setdefault(term, [])
            if len(ex) < 5:
                ex.append(rel)

    return counts, examples
