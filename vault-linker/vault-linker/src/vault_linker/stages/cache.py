from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

from vault_linker.core.hashing import sha256_text

CACHE_FILE = "state.json"

@dataclass
class CacheState:
    files: Dict[str, str]  # relpath -> sha256(normalized input)

def load_cache(cache_dir: Path) -> CacheState:
    p = cache_dir / CACHE_FILE
    if not p.exists():
        return CacheState(files={})
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
        files = obj.get("files", {})
        if not isinstance(files, dict):
            return CacheState(files={})
        return CacheState(files={str(k): str(v) for k, v in files.items()})
    except Exception:
        return CacheState(files={})

def save_cache(cache_dir: Path, state: CacheState) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / CACHE_FILE).write_text(
        json.dumps({"files": state.files}, indent=2, sort_keys=True),
        encoding="utf-8",
    )

def clear_cache(cache_dir: Path) -> None:
    p = cache_dir / CACHE_FILE
    if p.exists():
        p.unlink()

def should_process(relpath: str, normalized_input: str, cache: CacheState) -> bool:
    h = sha256_text(normalized_input)
    return cache.files.get(relpath) != h

def update_cache(relpath: str, normalized_input: str, cache: CacheState) -> None:
    cache.files[relpath] = sha256_text(normalized_input)
