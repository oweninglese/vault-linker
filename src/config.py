from pydantic import BaseModel, Field
from typing import List, Optional

class Config(BaseModel):
    allowed_encodings: List[str] = Field(default_factory=lambda: ["utf-8", "utf-8-sig", "cp1252"])
    md_glob: str = "**/*.md"
    dry_run: bool = True
    verbose: bool = False
    tags_file: str = "tags.csv"
    candidates_file: str = "tags_candidates.csv"
    cache_file: str = ".linker_cache.json"
    audit_log: str = ".linker_audit.log"
