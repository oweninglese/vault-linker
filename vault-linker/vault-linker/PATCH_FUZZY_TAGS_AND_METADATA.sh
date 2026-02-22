#!/usr/bin/env bash
set -euo pipefail

if [[ ! -d "src/vault_linker" ]] || [[ ! -f "pyproject.toml" ]]; then
  echo "ERROR: run from repo root (has src/vault_linker and pyproject.toml)."
  exit 1
fi

# -----------------------------
# 1) src/vault_linker/core/yaml.py
#    - omit keys with None / "null" / empty list where appropriate
# -----------------------------
cat > src/vault_linker/core/yaml.py <<'PY'
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import yaml

@dataclass(frozen=True)
class Frontmatter:
    data: Dict[str, Any]
    body: str

def split_frontmatter(md: str) -> Tuple[str, str]:
    if not md.startswith("---"):
        return "", md

    lines = md.splitlines(keepends=True)
    if not lines or lines[0].strip() != "---":
        return "", md

    end_idx = None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end_idx = i
            break
    if end_idx is None:
        raise ValueError("Malformed YAML frontmatter: missing closing '---'")

    yaml_text = "".join(lines[1:end_idx])
    body = "".join(lines[end_idx + 1 :])
    return yaml_text, body.lstrip("\n")

def parse_frontmatter(md: str) -> Frontmatter:
    yaml_text, body = split_frontmatter(md)
    if not yaml_text:
        return Frontmatter(data={}, body=body)
    try:
        data = yaml.safe_load(yaml_text) or {}
    except Exception as e:
        raise ValueError(f"Malformed YAML frontmatter: {e}") from e
    if not isinstance(data, dict):
        raise ValueError("Malformed YAML frontmatter: top-level must be a mapping")
    return Frontmatter(data=data, body=body)

def _nullish(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, str) and v.strip().lower() in {"null", "none", ""}:
        return True
    return False

def _clean_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Remove disgraceful placeholders:
    - drop None/"null"/"none"/"" fields
    - drop empty tags list ONLY if it was empty (we keep tags if non-empty)
    """
    out: Dict[str, Any] = {}
    for k, v in (data or {}).items():
        if _nullish(v):
            continue
        if k == "tags" and isinstance(v, list) and len(v) == 0:
            continue
        out[k] = v
    return out

def dump_frontmatter(data: Dict[str, Any]) -> str:
    data = _clean_data(data)
    y = yaml.safe_dump(
        data,
        sort_keys=True,
        allow_unicode=True,
        default_flow_style=False,
        width=88,
    ).strip()
    return f"---\n{y}\n---\n\n"
PY

# -----------------------------
# 2) src/vault_linker/stages/metadata.py  (new)
#    - salvage author/title/source/created from JSTOR-ish headers
# -----------------------------
cat > src/vault_linker/stages/metadata.py <<'PY'
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class Meta:
    title: Optional[str] = None
    author: Optional[str] = None
    source: Optional[str] = None
    created: Optional[str] = None  # year or ISO-like string

_RE_STABLE_URL = re.compile(r"(?im)^\s*Stable URL:\s*(\S+)\s*$")
_RE_SOURCE_URL = re.compile(r"(?im)^\s*(Source|URL):\s*(\S+)\s*$")
_RE_CHAPTER_TITLE = re.compile(r"(?im)^\s*Chapter Title:\s*(.+?)\s*$")
_RE_BOOK_TITLE = re.compile(r"(?im)^\s*Book Title:\s*(.+?)\s*$")
_RE_TITLE_LINE = re.compile(r"(?im)^\s*Title:\s*(.+?)\s*$")
_RE_AUTHOR = re.compile(r"(?im)^\s*(Author\(s\)|Chapter Author\(s\)|Author):\s*(.+?)\s*$")
_RE_PUBLISHED = re.compile(r"(?im)^\s*(Published|Publication Date):\s*(.+?)\s*$")
_RE_YEAR = re.compile(r"\b(18\d{2}|19\d{2}|20\d{2})\b")

def extract_metadata(title_fallback: str, body: str) -> Meta:
    """
    Uses only the first chunk to avoid scanning megabytes of PDFs.
    """
    head = "\n".join(body.splitlines()[:160])

    # Source URL preference: Stable URL > Source URL line
    source = None
    m = _RE_STABLE_URL.search(head)
    if m:
        source = m.group(1).strip()
    else:
        m2 = _RE_SOURCE_URL.search(head)
        if m2:
            source = m2.group(2).strip()

    # Title preference: Chapter Title > Title: > Book Title (fallback) > filename fallback
    title = None
    m = _RE_CHAPTER_TITLE.search(head)
    if m:
        title = m.group(1).strip()
    else:
        m = _RE_TITLE_LINE.search(head)
        if m:
            title = m.group(1).strip()
        else:
            m = _RE_BOOK_TITLE.search(head)
            if m:
                # Book title isn't the chapter/article title, but better than null
                title = m.group(1).strip()

    if not title:
        # As a last resort, use first non-empty line before boilerplate
        for ln in head.splitlines():
            ln = ln.strip()
            if not ln:
                continue
            if ln.lower().startswith(("jstor", "this content downloaded", "all use subject")):
                continue
            title = ln
            break

    if not title:
        title = title_fallback

    # Author
    author = None
    m = _RE_AUTHOR.search(head)
    if m:
        author = m.group(2).strip()

    # Created: year from Published line, else any year in header
    created = None
    m = _RE_PUBLISHED.search(head)
    if m:
        pub = m.group(2)
        ym = _RE_YEAR.search(pub)
        if ym:
            created = ym.group(1)
    if not created:
        ym = _RE_YEAR.search(head)
        if ym:
            created = ym.group(1)

    return Meta(title=title, author=author, source=source, created=created)
PY

# -----------------------------
# 3) src/vault_linker/stages/infer_tags.py
#    - fuzzy match registry tags:
#      * case-insensitive
#      * treat spaces and hyphens as equivalent separators
#      * tolerate straight/curly apostrophes in tag parts
# -----------------------------
cat > src/vault_linker/stages/infer_tags.py <<'PY'
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Iterable
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

def _fuzzy_pattern_for_tag(tag: str) -> re.Pattern:
    """
    Make a regex that matches tag tokens with these equivalences:
    - hyphen and spaces are interchangeable separators
    - case-insensitive
    - apostrophe in tag matches straight or curly apostrophe in text
    - safe boundaries (avoid matching inside longer alnum strings)
    """
    # Split on hyphen or whitespace in the tag vocabulary entry
    parts = [p for p in re.split(r"[\s\-]+", tag.strip()) if p]
    if not parts:
        # match nothing
        return re.compile(r"(?!)")

    def part_pat(p: str) -> str:
        # allow ' and ’ in text when vocab contains apostrophe-like chars
        # also tolerate them when absent by matching them literally only
        p = re.escape(p)
        p = p.replace(r"\'", r"['’]")
        return p

    sep = r"(?:[\s\-]+)"
    pat = sep.join(part_pat(p) for p in parts)

    # boundaries: not preceded/followed by word char or hyphen
    return re.compile(rf"(?<![\w\-]){pat}(?![\w\-])", flags=re.IGNORECASE)

def match_registry_tags(text: str, registry: List[str]) -> List[str]:
    found: List[str] = []
    for t in registry:
        if not t:
            continue
        if _fuzzy_pattern_for_tag(t).search(text):
            found.append(t)
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

    # If no registry, do not invent tags (curated-first policy)
    return TagResult(tags=base, candidates=[])
PY

# -----------------------------
# 4) src/vault_linker/pipeline/build.py
#    - apply metadata salvage
# -----------------------------
cat > src/vault_linker/pipeline/build.py <<'PY'
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Set, List

from vault_linker.config import BuildConfig
from vault_linker.logging import get_logger
from vault_linker.io.fs import ensure_dir, read_text_utf8, relpath_under, write_text_utf8
from vault_linker.core.tags import normalize_tag_list, read_tags_file
from vault_linker.core.yaml import dump_frontmatter
from vault_linker.stages.cache import load_cache, save_cache, should_process, update_cache, clear_cache
from vault_linker.stages.ingest import ingest
from vault_linker.stages.normalize import normalize
from vault_linker.stages.yaml_parse import parse
from vault_linker.stages.infer_tags import infer_tags
from vault_linker.stages.link_body import link_body
from vault_linker.stages.hubs import add_refs, build_hub_index, write_hubs
from vault_linker.stages.metadata import extract_metadata

log = get_logger()

@dataclass(frozen=True)
class BuildStats:
    total: int
    processed: int
    skipped_cache: int
    skipped_other: int
    yaml_errors: int

def build(cfg: BuildConfig) -> BuildStats:
    ensure_dir(cfg.output_vault)
    ensure_dir(cfg.cache_dir)

    if cfg.clear_cache and not cfg.dry_run:
        clear_cache(cfg.cache_dir)

    cache = load_cache(cfg.cache_dir)
    registry = read_tags_file(cfg.tags_file)
    if not registry:
        log.info(f"tag registry empty or missing: {cfg.tags_file}")

    hub_index = build_hub_index()
    items = ingest(cfg.input_vault)

    processed = 0
    skipped_cache = 0
    skipped_other = 0
    yaml_errors = 0

    for item in items:
        rel = relpath_under(cfg.input_vault, item.path)
        rel_str = str(rel.as_posix())
        filename_stem = item.path.stem

        try:
            raw = read_text_utf8(item.path)
        except UnicodeDecodeError:
            log.error(f"Non-UTF8 rejected: {rel_str}")
            skipped_other += 1
            continue

        normalized = normalize(raw)

        if not cfg.force and not should_process(rel_str, normalized, cache):
            skipped_cache += 1
            continue

        try:
            fm = parse(normalized)
        except ValueError as e:
            log.error(f"YAML error in {rel_str}: {e}")
            yaml_errors += 1
            skipped_other += 1
            continue

        # ---- Metadata salvage ----
        meta = extract_metadata(filename_stem, fm.body)

        # Only set fields if missing/nullish
        def nullish(v) -> bool:
            return v is None or (isinstance(v, str) and v.strip().lower() in {"null", "none", ""})

        if nullish(fm.data.get("title", None)) and meta.title:
            fm.data["title"] = meta.title
        if nullish(fm.data.get("author", None)) and meta.author:
            fm.data["author"] = meta.author
        if nullish(fm.data.get("source", None)) and meta.source:
            fm.data["source"] = meta.source
        if nullish(fm.data.get("created", None)) and meta.created:
            fm.data["created"] = meta.created

        title_for_tagging = str(fm.data.get("title", filename_stem))

        # ---- Tagging from curated registry ----
        res = infer_tags(
            title=title_for_tagging,
            body=fm.body,
            registry=registry,
            yaml_tags=fm.data.get("tags", None),
            vocab_only=cfg.vocab_only,
        )
        fm.data["tags"] = normalize_tag_list(res.tags)

        # ---- Linking: first occurrence per paragraph per tag ----
        body_out = fm.body
        if not cfg.no_links and fm.data["tags"]:
            body_out = link_body(body_out, fm.data["tags"])

        # Optional hashtags tail
        hashtags = " ".join(f"#{t}" for t in fm.data["tags"])
        if hashtags:
            body_out = body_out.rstrip("\n") + "\n\n" + hashtags + "\n"

        out_text = dump_frontmatter(fm.data) + body_out.lstrip("\n")
        out_path = (cfg.output_vault / rel).resolve()

        if cfg.dry_run:
            log.info(f"[dry-run] would write: {rel_str}")
        else:
            write_text_utf8(out_path, out_text)
            update_cache(rel_str, normalized, cache)

        add_refs(hub_index, rel_str, fm.data["tags"])
        processed += 1

    if not cfg.dry_run:
        write_hubs(cfg.output_vault, hub_index)
        save_cache(cfg.cache_dir, cache)

    return BuildStats(
        total=len(items),
        processed=processed,
        skipped_cache=skipped_cache,
        skipped_other=skipped_other,
        yaml_errors=yaml_errors,
    )
PY

echo "OK: patched fuzzy tag matching + metadata salvage + null cleanup"
