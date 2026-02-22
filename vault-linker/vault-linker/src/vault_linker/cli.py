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
        emit_candidates=None,
    )

    stats = build(cfg)
    log.info(
        "done: total=%d processed=%d skipped_cache=%d skipped_other=%d yaml_errors=%d vocab_only=%s tags_file=%s output=%s",
        stats.total, stats.processed, stats.skipped_cache, stats.skipped_other,
        stats.yaml_errors, cfg.vocab_only, cfg.tags_file, cfg.output_vault
    )
    return 0
