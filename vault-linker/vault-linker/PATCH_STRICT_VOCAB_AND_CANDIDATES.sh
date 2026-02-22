#!/usr/bin/env bash
set -euo pipefail

if [[ ! -d "src/vault_linker" ]] || [[ ! -f "pyproject.toml" ]]; then
  echo "ERROR: run from repo root."
  exit 1
fi

# -----------------------------
# src/vault_linker/config.py
# -----------------------------
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
    dry_run: bool = False
    force: bool = False
    clear_cache: bool = False
    vocab_only: bool = False          # only write YAML tags that exist in tags_file registry
    emit_candidates: Path | None = None  # write candidate tags here (no auto-append)
PY

# -----------------------------
# src/vault_linker/stages/infer_tags.py
#   Strict behavior:
#   - If registry exists and vocab_only=True:
#       tags = yaml_tags ∪ {registry tags found in (title+body)}   (exact match, safe boundaries)
#       candidates = [] (we don't invent)
#   - If registry exists and vocab_only=False:
#       tags = yaml_tags ∪ {registry tags found in (title+body)}
#       candidates = "new-ish" signals from title/filename/body, BUT heavily gated (optional use)
#   - If registry missing/empty:
#       tags = yaml_tags only (we do not invent tags)
#       candidates = gated suggestions (for emit only)
# -----------------------------
cat > src/vault_linker/stages/infer_tags.py <<'PY'
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Set, Iterable
import re

from vault_linker.core.tags import canonicalize_tag, normalize_tag_list

@dataclass(frozen=True)
class TagResult:
    tags: List[str]            # tags written to YAML
    candidates: List[str]      # suggestions for review (never auto-appended)

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

def _safe_boundary_pattern(tag: str) -> re.Pattern:
    # For hyphenated tags, treat hyphen as part of word.
    # Ensure we don't match inside AAAAAAAAAA or COVID19X etc.
    return re.compile(rf"(?<![\w\-]){re.escape(tag)}(?![\w\-])")

def match_registry_tags(text: str, registry: List[str]) -> List[str]:
    """
    Deterministic: returns registry tags that appear in text using safe boundaries.
    Performance: O(|registry| * search). For very large registries you may want an index/trie later.
    """
    found: List[str] = []
    for t in registry:
        if not t:
            continue
        if _safe_boundary_pattern(t).search(text):
            found.append(t)
    return _uniq_ci(found)

# --- Candidate gating for optional discovery/export only ---

_ALPHA = re.compile(r"[A-Za-z]")
_VALID = re.compile(r"^[A-Za-z][A-Za-z\-]{2,}$")  # letters/hyphen only, length>=3, starts with letter

def gated_candidates_from_text(text: str, *, max_n: int = 200) -> List[str]:
    """
    Very conservative candidates:
    - Only letters/hyphen
    - Must start with a letter
    - No digits, no underscores, no base64-ish, no long screaming tokens
    """
    toks = re.findall(r"\b[A-Za-z][A-Za-z\-]{2,}\b", text or "")
    out: List[str] = []
    for t in toks:
        c = canonicalize_tag(t)
        if not c:
            continue
        if len(c) > 40:        # hard stop on long junk
            continue
        if not _VALID.match(c): # excludes digit noise like A0109036, 2020, A549, etc
            continue
        out.append(c)
        if len(out) >= max_n:
            break
    return _uniq_ci(out)

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

        if vocab_only:
            return TagResult(tags=tags_out, candidates=[])

        # Non-vocab-only: still do NOT auto-add candidates; emit for review only.
        cand = gated_candidates_from_text(title, max_n=50)
        return TagResult(tags=tags_out, candidates=cand)

    # No registry: do not invent YAML tags; only emit conservative candidates
    cand = gated_candidates_from_text(title, max_n=50)
    return TagResult(tags=base, candidates=cand)
PY

# -----------------------------
# src/vault_linker/pipeline/build.py
#   - remove auto-append behavior
#   - if emit_candidates is set: write unique candidates observed
# -----------------------------
cat > src/vault_linker/pipeline/build.py <<'PY'
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Set

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

log = get_logger()

@dataclass(frozen=True)
class BuildStats:
    total: int
    processed: int
    skipped_cache: int
    skipped_other: int
    yaml_errors: int
    emitted_candidates: int

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

    cand_seen_ci: Set[str] = set()
    cand_out: List[str] = []

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

        res = infer_tags(
            title=title,
            body=fm.body,
            registry=registry,
            yaml_tags=fm.data.get("tags", None),
            vocab_only=cfg.vocab_only,
        )

        fm.data["tags"] = normalize_tag_list(res.tags)

        if cfg.emit_candidates is not None:
            for t in res.candidates:
                k = t.casefold()
                if k not in cand_seen_ci:
                    cand_seen_ci.add(k)
                    cand_out.append(t)

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
        write_hubs(cfg.output_vault, hub_index)
        save_cache(cfg.cache_dir, cache)

        if cfg.emit_candidates is not None:
            cfg.emit_candidates.parent.mkdir(parents=True, exist_ok=True)
            cfg.emit_candidates.write_text("\n".join(cand_out) + ("\n" if cand_out else ""), encoding="utf-8")

    return BuildStats(
        total=len(items),
        processed=processed,
        skipped_cache=skipped_cache,
        skipped_other=skipped_other,
        yaml_errors=yaml_errors,
        emitted_candidates=len(cand_out),
    )
PY

# -----------------------------
# src/vault_linker/cli.py
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
    b.add_argument("--vocab-only", action="store_true", help="Only tag using entries present in tags registry (curated mode)")
    b.add_argument("--emit-candidates", default=None, help="Write conservative candidate tags to this file (for review)")
    b.add_argument("--no-links", action="store_true", help="Disable body linking")
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

    emit = Path(args.emit_candidates).expanduser().resolve() if args.emit_candidates else None

    cfg = BuildConfig(
        input_vault=input_vault,
        output_vault=output_vault,
        cache_dir=cache_dir,
        tags_file=tags_file,
        no_links=bool(args.no_links),
        dry_run=bool(args.dry_run),
        force=bool(args.force),
        clear_cache=bool(args.clear_cache),
        vocab_only=bool(args.vocab_only),
        emit_candidates=emit,
    )

    stats = build(cfg)
    log.info(
        "done: total=%d processed=%d skipped_cache=%d skipped_other=%d yaml_errors=%d emitted_candidates=%d vocab_only=%s tags_file=%s",
        stats.total, stats.processed, stats.skipped_cache, stats.skipped_other,
        stats.yaml_errors, stats.emitted_candidates, cfg.vocab_only, cfg.tags_file
    )
    return 0
PY

echo "OK: strict vocab matching + emit candidates (no auto-learn)"
