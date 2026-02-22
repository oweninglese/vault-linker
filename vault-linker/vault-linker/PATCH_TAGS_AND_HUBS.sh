
#!/usr/bin/env bash
set -euo pipefail

# Must be run from repo root
if [[ ! -d "src/vault_linker" ]] || [[ ! -f "pyproject.toml" ]]; then
  echo "ERROR: run from the repo root (where src/vault_linker and pyproject.toml exist)."
  exit 1
fi

# -----------------------------
# 1) src/vault_linker/stages/infer_tags.py
#   - infer tags from title + filename + body
#   - if registry exists: only accept candidates already in registry unless learn_tags
#   - if registry empty + learn_tags: accept candidates (bounded)
# -----------------------------
cat > src/vault_linker/stages/infer_tags.py <<'PY'
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Set, Optional

from vault_linker.core.tags import (
    infer_from_body,
    infer_from_title,
    canonicalize_tag,
    normalize_tag_list,
)

@dataclass(frozen=True)
class TagResult:
    tags: List[str]            # tags to write into YAML for this doc
    new_candidates: List[str]  # tags that are not in registry (for learning)

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
    """
    Use filename as a weak signal, but it saves your 'title: null' imports.
    Split on common separators and canonicalize.
    """
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
    max_auto_tags: int = 25,
) -> TagResult:
    """
    Behavior:
    - Always normalize YAML tags.
    - Always infer candidates (title + filename + body).
    - If registry is non-empty:
        - tags written to YAML = (yaml tags) ∪ (candidates ∩ registry)
        - new_candidates = candidates \ registry
        - if learn_tags: new_candidates can be appended to registry file (but still bounded)
    - If registry is empty:
        - tags written to YAML = yaml tags
        - if learn_tags: tags written to YAML also include top candidates (bounded)
    """
    base = normalize_tag_list(yaml_tags)

    reg_ci: Set[str] = {t.casefold() for t in registry}
    base_ci: Set[str] = {t.casefold() for t in base}

    candidates: List[str] = []
    candidates.extend(infer_from_title(title))
    candidates.extend(infer_from_filename(filename_stem))
    candidates.extend(infer_from_body(body))
    candidates = _uniq_ci(candidates)

    if registry:
        # Only accept candidates already in registry for YAML tags (prevents explosion)
        accepted = [t for t in candidates if t.casefold() in reg_ci]
        # candidates not in registry are "learnable"
        new = [t for t in candidates if t.casefold() not in reg_ci]
        tags_out = _uniq_ci(base + accepted)

        # If learn_tags is on, we still only *write* registry-known tags to YAML by default.
        # Learning affects the registry file, not YAML tag inflation.
        return TagResult(tags=tags_out, new_candidates=new)

    # No registry available:
    if learn_tags:
        # Use candidates directly, but bounded
        bounded = candidates[:max_auto_tags]
        tags_out = _uniq_ci(base + bounded)
        new = bounded  # all "new" because registry empty
        return TagResult(tags=tags_out, new_candidates=new)

    # No registry + not learning: only YAML tags
    return TagResult(tags=base, new_candidates=[])
PY

# -----------------------------
# 2) src/vault_linker/pipeline/build.py
#   - fallback title when null/missing using filename stem
#   - pass filename stem into infer_tags
#   - only link tags that are in YAML tags list (after inference)
# -----------------------------
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
from vault_linker.stages.cache import load_cache, save_cache, should_process, update_cache
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
    skipped: int
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

    cache = load_cache(cfg.cache_dir)

    # Registry may be empty or missing; read_tags_file handles missing file
    registry = read_tags_file(cfg.tags_file)
    if not registry:
        log.info(f"tag registry empty or missing: {cfg.tags_file}")

    hub_index = build_hub_index()
    items = ingest(cfg.input_vault)

    processed = 0
    skipped = 0
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
            skipped += 1
            continue

        normalized = normalize(raw)

        if not should_process(rel_str, normalized, cache):
            skipped += 1
            continue

        try:
            fm = parse(normalized)
        except ValueError as e:
            log.error(f"YAML error in {rel_str}: {e}")
            yaml_errors += 1
            skipped += 1
            continue

        # Title fallback: many PDF-derived docs have title: null
        title_val = fm.data.get("title", None)
        if _is_nullish(title_val):
            fm.data["title"] = filename_stem
            title = filename_stem
        else:
            title = str(title_val)

        yaml_tags = fm.data.get("tags", None)

        tag_res = infer_tags(
            title=title,
            filename_stem=filename_stem,
            body=fm.body,
            registry=registry,
            yaml_tags=yaml_tags,
            learn_tags=cfg.learn_tags,
            max_auto_tags=25,
        )

        # Enforce YAML tag list normalization
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

        # Hubs are driven by the YAML tags list (now filled deterministically)
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
        skipped=skipped,
        yaml_errors=yaml_errors,
        learned_tags=learned_count,
    )
PY

# -----------------------------
# 3) src/vault_linker/cli.py
#   - make tags-file behavior explicit
#   - if tags file directory doesn't exist, still works
# -----------------------------
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
    b.add_argument("--no-links", action="store_true", help="Disable body linking")
    b.add_argument("--learn-tags", action="store_true", help="Learn inferred tags into tag registry (bounded)")
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
    )

    stats = build(cfg)
    log.info(
        f"done: total={stats.total} processed={stats.processed} skipped={stats.skipped} "
        f"yaml_errors={stats.yaml_errors} learned_tags={stats.learned_tags} tags_file={cfg.tags_file}"
    )
    return 0
PY

echo "OK: patched infer+pipeline+cli"
echo "Next: pip install -e ."
