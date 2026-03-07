from __future__ import annotations

from pathlib import Path
from typing import List
from pydantic import BaseModel, Field


class Config(BaseModel):
    allowed_encodings: List[str] = Field(
        default_factory=lambda: ["utf-8", "utf-8-sig", "cp1252"]
    )

    dry_run: bool = True
    verbose: bool = False
    link_bodies: bool = True
    repair_frontmatter: bool = False
    allow_missing_tagsfile: bool = False

    # 1 = once per file
    # 2 = once per paragraph (future)
    # 3 = all occurrences (future)
    linkify_mode: int = 1

    tags_file: Path | None = None
    candidates_file: Path = Path(".vault-linker/tag_candidates.txt")

    ignore_dirs: List[str] = Field(
        default_factory=lambda: [
            ".git",
            ".obsidian",
            ".trash",
            ".vault-linker",
            "node_modules",
            "__pycache__",
            ".venv",
        ]
    )

    min_term_len: int = 3

    vault: Path
    hub_dir: Path | None = None

    hub_marker_start: str = "<!-- VAULT_LINKER_HUB_START -->"
    hub_marker_end: str = "<!-- VAULT_LINKER_HUB_END -->"

    def hub_path_for(self, term: str) -> Path:
        base = self.hub_dir if self.hub_dir is not None else self.vault
        safe_term = (
            term.strip()
            .replace("/", "-")
            .replace("\\", "-")
        )
        return base / f"{safe_term}.md"
