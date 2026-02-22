#!/usr/bin/env bash
set -euo pipefail

if [[ ! -d "src/vault_linker" ]] || [[ ! -f "pyproject.toml" ]]; then
  echo "ERROR: run from repo root."
  exit 1
fi

# 1) config.py
cat > src/vault_linker/config.py <<'PY'
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class BuildConfig:
    input_vault: Path
    output_vault: Path
    cache_dir: Path
    tags_file: Path
    no_links: bool = False
    learn_tags: bool = False
    dry_run: bool = False
    force: bool = False
    clear_cache: bool = False
    vocab_only: bool = False  # only write YAML tags that exist in tags_file registry
PY

# 2) stages/infer_tags.py
cat > src/vault_linker/stages/infer_tags.py <<'PY'
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Set

from vault_linker.core.tags import (
    infer_from_body,
    infer_from_title,
    canonicalize_tag,
    normalize_tag_list,
)

@dataclass(frozen=True)
class TagResult:
    tags: List[str]            # tags written to YAML
    new_candidates: List[str]  # out-of-vocab candidates (for optional learning)

def _uniq_ci(xs: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for x in xs:
        if not x:
            continue
        k = x.casefold()
        if k not in seen:
            out.append(x)
            seen.add(k)
    return out

def infer_from_filename(stem: str) -> List[str]:
    if not stem:
        return []
    parts: List[str] = []
    for chunk in stem.replace("_", " ").replace("-", " ").split():
        c = canonicalize_tag(chunk)
        if c and len(c) >= 3:
            parts.append(c)
    return _uniq_ci(parts)

def infer_tags(
    *,
    title: str,
    filename_stem: str,
    body: str,
    registry: List[str],
    yaml_tags,
    learn_tags: bool,
    vocab_only: bool,
    max_auto_tags: int = 25,
) -> TagResult:
    base = normalize_tag_list(yaml_tags)
    reg_ci: Set[str] = {t.casefold() for t in registry}

    candidates: List[str] = []
    candidates.extend(infer_from_title(title))
    candidates.extend(infer_from_filename(filename_stem))
    candidates.extend(infer_from_body(body))
    candidates = _uniq_ci(candidates)

    if registry:
        accepted = [t for t in candidates if t.casefold() in reg_ci]
        new = [t for t in candidates if t.casefold() not in reg_ci]

        # vocab_only means: YAML tags = YAML tags ∪ (candidates ∩ registry)
        # (never inject out-of-vocab tags into YAML)
        tags_out = _uniq_ci(base + accepted)

        # learn_tags controls whether we return new candidates for appending to registry
        return TagResult(tags=tags_out, new_candidates=(new if learn_tags else []))

    # No registry present (or empty)
    if vocab_only:
        # if user demands vocab-only but registry empty, we do nothing beyond YAML
        return TagResult(tags=base, new_candidates=[])

    if learn_tags:
        bounded = candidates[:max_auto_tags]
        tags_out = _uniq_ci(base + bounded)
        return TagResult(tags=tags_out, new_candidates=bounded)

    return TagResult(tags=base, new_candidates=[])
PY

# 3) pipeline/build.py (pass vocab_only + change learned count)
cat > src/vault_linker/pipeline/build.py <<'PY'
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

from vault_linker.config import BuildConfig
from vault_linker.logging import get_logger
from vault_linker.io.fs import ensure_dir, read_text_utf8, relpath_under, write_text_utf8
from vault_linker.core.tags import append_tags_file, normalize_tag_list, read_tags_file
from vault_linker.core.yaml import dump_frontmatter
from vault_linker.stages.cache import load_cache, save_cache, should_process, update_cache, clear_cache
from vault_linker.stages.ingest import ingest
from vault_linker.stages.normalize import normalize
from vault_linker.stages.yaml_parse import parse
from vault_linker.stages.infer_tags import infer_tags
from vault_linker.stages.link_body import link_body
from vault_linker.stages.hubs import add_refs, build_hub_index, write_hubs

log = get_logger()

@dataclass(frozen=True)
class BuildStats:
    total: int
    processed: int
    skipped_cache: int
    skipped_other: int
    yaml_errors: int
    learned_tags: int

def _is_nullish(x) -> bool:
    if x is None:
        return True
    if isinstance(x, str) and x.strip().lower() in {"null", "none", ""}:
        return True
    return False

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
    learned: List[str] = []

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

        title_val = fm.data.get("title", None)
        if _is_nullish(title_val):
            fm.data["title"] = filename_stem
            title = filename_stem
        else:
            title = str(title_val)

        tag_res = infer_tags(
            title=title,
            filename_stem=filename_stem,
            body=fm.body,
            registry=registry,
            yaml_tags=fm.data.get("tags", None),
            learn_tags=cfg.learn_tags,
            vocab_only=cfg.vocab_only,
            max_auto_tags=25,
        )

        fm.data["tags"] = normalize_tag_list(tag_res.tags)

        if cfg.learn_tags and tag_res.new_candidates:
            learned.extend(tag_res.new_candidates)

        body_out = fm.body
        if not cfg.no_links and fm.data["tags"]:
            body_out = link_body(body_out, fm.data["tags"])

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
        if cfg.learn_tags and learned:
            append_tags_file(cfg.tags_file, learned)
        write_hubs(cfg.output_vault, hub_index)
        save_cache(cfg.cache_dir, cache)

    learned_count = len({t.casefold(): t for t in learned})
    return BuildStats(
        total=len(items),
        processed=processed,
        skipped_cache=skipped_cache,
        skipped_other=skipped_other,
        yaml_errors=yaml_errors,
        learned_tags=learned_count,
    )
PY

# 4) cli.py add flag
cat > src/vault_linker/cli.py <<'PY'
from __future__ import annotations

import argparse
from pathlib import Path

from vault_linker.config import BuildConfig
from vault_linker.logging import get_logger
from vault_linker.pipeline.build import build

log = get_logger()

def _default_tags_file() -> Path:
    return Path.home() / "library" / "tags" / "TAGS.txt"

def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="vault-linker")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="Build an output vault from an input vault")
    b.add_argument("vault", help="Path to input vault")
    b.add_argument("--output", required=True, help="Path to output vault")
    b.add_argument("--cache-dir", default=None, help="Cache dir (default: <output>/.vault_linker_cache)")
    b.add_argument("--tags-file", default=None, help="Tag registry file (default: ~/library/tags/TAGS.txt; fallback: <vault>/TAGS.txt)")
    b.add_argument("--vocab-only", action="store_true", help="Only use tags found in tags registry (recommended for curated TAGS.txt)")
    b.add_argument("--no-links", action="store_true", help="Disable body linking")
    b.add_argument("--learn-tags", action="store_true", help="Append out-of-vocab candidates to tag registry (explicit)")
    b.add_argument("--force", action="store_true", help="Reprocess all files (ignore cache)")
    b.add_argument("--clear-cache", action="store_true", help="Clear cache state before running")
    b.add_argument("--dry-run", action="store_true", help="No writes; report actions")

    args = p.parse_args(argv)

    input_vault = Path(args.vault).expanduser().resolve()
    output_vault = Path(args.output).expanduser().resolve()
    cache_dir = Path(args.cache_dir).expanduser().resolve() if args.cache_dir else (output_vault / ".vault_linker_cache")

    tags_file = Path(args.tags_file).expanduser().resolve() if args.tags_file else _default_tags_file()
    if args.tags_file is None and not tags_file.exists():
        tags_file = input_vault / "TAGS.txt"

    cfg = BuildConfig(
        input_vault=input_vault,
        output_vault=output_vault,
        cache_dir=cache_dir,
        tags_file=tags_file,
        no_links=bool(args.no_links),
        learn_tags=bool(args.learn_tags),
        dry_run=bool(args.dry_run),
        force=bool(args.force),
        clear_cache=bool(args.clear_cache),
        vocab_only=bool(args.vocab_only),
    )

    stats = build(cfg)
    log.info(
        "done: total=%d processed=%d skipped_cache=%d skipped_other=%d yaml_errors=%d learned_tags=%d vocab_only=%s tags_file=%s",
        stats.total, stats.processed, stats.skipped_cache, stats.skipped_other,
        stats.yaml_errors, stats.learned_tags, cfg.vocab_only, cfg.tags_file
    )
    return 0
PY

echo "OK: added --vocab-only (curated registry mode)"
