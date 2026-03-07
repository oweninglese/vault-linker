from __future__ import annotations

import argparse
from pathlib import Path

from .config import Config
from .runner import run


def main() -> int:
    parser = argparse.ArgumentParser(prog="vault-linker")
    sub = parser.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run")
    r.add_argument("vault", type=Path)
    r.add_argument("--tagfile", type=Path, default=None)
    r.add_argument("--allow-missing-tagsfile", action="store_true")
    r.add_argument("--hub-dir", type=str, default="")
    r.add_argument("--db", type=Path, default=None)
    r.add_argument("--no-link-bodies", action="store_true")
    r.add_argument("--dry-run", action="store_true")
    r.add_argument("--reindex", action="store_true")
    r.add_argument("--verbose", action="store_true")
    r.add_argument("--repair-frontmatter", action="store_true")

    r.add_argument("--discover", action="store_true")
    r.add_argument("--discover-min-count", type=int, default=3)
    r.add_argument("--discover-out", type=Path, default=None)
    r.add_argument("--discover-acronyms", action="store_true")

    args = parser.parse_args()

    if args.cmd != "run":
        parser.error("Only the 'run' command is supported.")

    vault = args.vault.expanduser().resolve()
    if not vault.exists() or not vault.is_dir():
        raise SystemExit(f"Vault does not exist or is not a directory: {vault}")

    tagfile = None
    if args.tagfile is not None:
        tagfile = args.tagfile.expanduser().resolve()
        if not tagfile.exists() or not tagfile.is_file():
            raise SystemExit(f"Tag file does not exist or is not a file: {tagfile}")
    elif not args.allow_missing_tagsfile:
        raise SystemExit("Missing required --tagfile (or use --allow-missing-tagsfile)")

    hub_dir_path = (vault / args.hub_dir.strip()).resolve() if args.hub_dir.strip() else None
    db_path = args.db.expanduser().resolve() if args.db else (vault / ".vault-linker" / "index.sqlite")

    cfg = Config(
        vault=vault,
        tags_file=tagfile,
        hub_dir=hub_dir_path,
        link_bodies=not args.no_link_bodies,
        dry_run=args.dry_run,
        verbose=args.verbose,
        repair_frontmatter=args.repair_frontmatter,
        allow_missing_tagsfile=args.allow_missing_tagsfile,
    )

    stats = run(
        cfg,
        db_path=db_path,
        dry_run=args.dry_run,
        verbose=args.verbose,
        reindex=args.reindex,
        discover=args.discover,
        discover_min_count=args.discover_min_count,
        discover_out=args.discover_out,
        discover_acronyms=args.discover_acronyms,
    )

    print(
        f"[vault-linker] terms={stats.terms} scanned={stats.scanned} processed={stats.processed} "
        f"wrote_notes={stats.wrote_notes} inserted_links={stats.inserted_links} "
        f"hubs_updated={stats.hubs_updated} hubs_scrubbed={stats.hubs_scrubbed} "
        f"candidates_written={stats.candidates_written} diagnostics={stats.diagnostics}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
