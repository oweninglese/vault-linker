
from __future__ import annotations

import argparse
from pathlib import Path

from .config import Config
from .runner import run


def main() -> int:
    p = argparse.ArgumentParser(prog="vault-linker")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run")
    r.add_argument("vault", type=Path)
    r.add_argument("--tagfile", type=Path, required=True)
    r.add_argument("--hub-dir", type=str, default="")
    r.add_argument("--db", type=Path, default=None)
    r.add_argument("--no-link-bodies", action="store_true")
    r.add_argument("--dry-run", action="store_true")
    r.add_argument("--reindex", action="store_true")
    r.add_argument("--verbose", action="store_true")

    # Discovery mode
    r.add_argument("--discover", action="store_true", help="Infer candidate tags and write a review file.")
    r.add_argument("--discover-min-count", type=int, default=3, help="Minimum vault-wide count for candidates.")
    r.add_argument("--discover-out", type=Path, default=None, help="Output file for candidates (default: <vault>/.vault-linker/tag_candidates.txt)")
    r.add_argument("--discover-acronyms", action="store_true", help="Include acronyms like WHO/UN (noisier).")

    args = p.parse_args()
    vault = args.vault.expanduser().resolve()

    hub_dir = args.hub_dir.strip()
    hub_dir_path = Path(hub_dir) if hub_dir else None

    db_path = args.db or (vault / ".vault-linker" / "index.sqlite")

    cfg = Config(
        vault=vault,
        tagfile=args.tagfile.expanduser().resolve(),
        hub_dir=hub_dir_path,
        link_bodies=not args.no_link_bodies,
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
        f"candidates_written={stats.candidates_written}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
