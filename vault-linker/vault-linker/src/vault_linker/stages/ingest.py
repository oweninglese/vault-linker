from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

from vault_linker.io.fs import iter_md_files

@dataclass(frozen=True)
class IngestItem:
    path: Path

def ingest(vault_root: Path) -> List[IngestItem]:
    return [IngestItem(path=p) for p in iter_md_files(vault_root)]
