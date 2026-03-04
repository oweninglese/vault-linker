
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import os
import sqlite3
from typing import Iterable, Iterator


@dataclass(frozen=True)
class FileRec:
    path: Path
    rel: str
    mtime_ns: int
    size: int
    sha1: str


def _sha1_bytes(data: bytes) -> str:
    h = hashlib.sha1()
    h.update(data)
    return h.hexdigest()


def ensure_db(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(db_path))
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS files (
          rel TEXT PRIMARY KEY,
          mtime_ns INTEGER NOT NULL,
          size INTEGER NOT NULL,
          sha1 TEXT NOT NULL
        )
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS mentions (
          rel TEXT NOT NULL,
          term TEXT NOT NULL,
          PRIMARY KEY (rel, term)
        )
        """
    )
    con.execute("CREATE INDEX IF NOT EXISTS idx_mentions_term ON mentions(term)")
    return con


def iter_markdown_files(vault: Path, ignore_dirs: Iterable[str]) -> Iterator[Path]:
    ignore = set(ignore_dirs)
    for root, dirs, files in os.walk(vault):
        # Prune directories in-place for speed
        dirs[:] = [d for d in dirs if d not in ignore]
        for fn in files:
            if fn.endswith(".md"):
                yield Path(root) / fn


def stat_file(path: Path) -> tuple[int, int]:
    st = path.stat()
    return st.st_mtime_ns, st.st_size


def read_and_hash(path: Path) -> tuple[str, str]:
    b = path.read_bytes()
    return b.decode("utf-8", errors="replace"), _sha1_bytes(b)


def get_cached_file(con: sqlite3.Connection, rel: str) -> tuple[int, int, str] | None:
    row = con.execute("SELECT mtime_ns, size, sha1 FROM files WHERE rel = ?", (rel,)).fetchone()
    if not row:
        return None
    return int(row[0]), int(row[1]), str(row[2])


def upsert_file(con: sqlite3.Connection, rel: str, mtime_ns: int, size: int, sha1: str) -> None:
    con.execute(
        "INSERT INTO files(rel, mtime_ns, size, sha1) VALUES(?,?,?,?) "
        "ON CONFLICT(rel) DO UPDATE SET mtime_ns=excluded.mtime_ns, size=excluded.size, sha1=excluded.sha1",
        (rel, mtime_ns, size, sha1),
    )


def replace_mentions(con: sqlite3.Connection, rel: str, terms: set[str]) -> None:
    con.execute("DELETE FROM mentions WHERE rel = ?", (rel,))
    con.executemany("INSERT OR IGNORE INTO mentions(rel, term) VALUES(?,?)", [(rel, t) for t in sorted(terms)])


def get_backlinks(con: sqlite3.Connection, term: str) -> list[str]:
    rows = con.execute("SELECT rel FROM mentions WHERE term = ? ORDER BY rel", (term,)).fetchall()
    return [str(r[0]) for r in rows]
