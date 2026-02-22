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
