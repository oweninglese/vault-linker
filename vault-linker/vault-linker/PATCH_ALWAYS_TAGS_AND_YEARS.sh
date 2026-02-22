#!/usr/bin/env bash
set -euo pipefail

if [[ ! -d "src/vault_linker" ]] || [[ ! -f "pyproject.toml" ]]; then
  echo "ERROR: run from repo root."
  exit 1
fi

# -----------------------------
# 1) Always keep 'tags' key in YAML (even when empty)
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
    Remove null placeholders, but ALWAYS keep 'tags' key (even if empty list).
    """
    out: Dict[str, Any] = {}
    for k, v in (data or {}).items():
        if k != "tags" and _nullish(v):
            continue
        out[k] = v

    # ensure tags key exists
    if "tags" not in out or out["tags"] is None:
        out["tags"] = []
    if isinstance(out["tags"], tuple):
        out["tags"] = list(out["tags"])
    if not isinstance(out["tags"], list):
        out["tags"] = [str(out["tags"])]

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
# 2) Add year extraction stage (new)
# -----------------------------
cat > src/vault_linker/stages/years.py <<'PY'
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
PY

# -----------------------------
# 3) Pipeline: always set tags list, add registry matches + year tags, then link
# -----------------------------
cat > src/vault_linker/pipeline/build.py <<'PY'
from __future__ import annotations

from dataclasses import dataclass

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
from vault_linker.stages.years import extract_year_tags

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

    def nullish(v) -> bool:
        return v is None or (isinstance(v, str) and v.strip().lower() in {"null", "none", ""})

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
        if nullish(fm.data.get("title", None)) and meta.title:
            fm.data["title"] = meta.title
        if nullish(fm.data.get("author", None)) and meta.author:
            fm.data["author"] = meta.author
        if nullish(fm.data.get("source", None)) and meta.source:
            fm.data["source"] = meta.source
        if nullish(fm.data.get("created", None)) and meta.created:
            fm.data["created"] = meta.created

        # Ensure tags key exists even before inference
        fm.data["tags"] = normalize_tag_list(fm.data.get("tags", None))

        title_for_tagging = str(fm.data.get("title", filename_stem))

        # ---- Curated registry tag matches (space/hyphen + case-insensitive already) ----
        res = infer_tags(
            title=title_for_tagging,
            body=fm.body,
            registry=registry,
            yaml_tags=fm.data.get("tags", None),
            vocab_only=cfg.vocab_only,
        )
        tags = normalize_tag_list(res.tags)

        # ---- Add year tags from body (independent of registry) ----
        year_tags = extract_year_tags(fm.body)
        tags = normalize_tag_list(tags + year_tags)

        fm.data["tags"] = tags  # always present now

        # ---- Linking: first occurrence per paragraph per tag ----
        body_out = fm.body
        if not cfg.no_links and tags:
            body_out = link_body(body_out, tags)

        out_text = dump_frontmatter(fm.data) + body_out.lstrip("\n")
        out_path = (cfg.output_vault / rel).resolve()

        if cfg.dry_run:
            log.info(f"[dry-run] would write: {rel_str}")
        else:
            write_text_utf8(out_path, out_text)
            update_cache(rel_str, normalized, cache)

        add_refs(hub_index, rel_str, tags)
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

echo "OK: tags always present + year tags added"
