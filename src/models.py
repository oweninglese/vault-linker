from pydantic import BaseModel, Field, ConfigDict
from typing import Dict, List, Optional, Any
from diagnostics import Diagnostic
from rich.table import Table

class BaseVaultModel(BaseModel):
    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

class Document(BaseVaultModel):
    path: str
    title: str
    body: str
    encoding: str
    frontmatter: Dict[str, Any] = Field(default_factory=dict)
    raw_frontmatter: Optional[str] = None

class VaultIndex(BaseVaultModel):
    documents: Dict[str, Document] = Field(default_factory=dict)
    diagnostics: List[Diagnostic] = Field(default_factory=list)

class ApplyReport(BaseVaultModel):
    files_scanned: int = 0
    files_modified: int = 0
    links_created: int = 0
    errors: List[Diagnostic] = Field(default_factory=list)

    def as_rich_table(self) -> Table:
        t = Table(title="Vault-Linker 3.0 Summary", header_style="bold cyan")
        t.add_column("Metric")
        t.add_column("Value", justify="right")
        t.add_row("Files Scanned", str(self.files_scanned))
        t.add_row("Modified", str(self.files_modified), style="green" if self.files_modified > 0 else "")
        t.add_row("Links Created", str(self.links_created), style="bold")
        return t
