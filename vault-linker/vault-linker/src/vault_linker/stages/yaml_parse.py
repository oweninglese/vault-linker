from __future__ import annotations
from vault_linker.core.yaml import Frontmatter, parse_frontmatter
def parse(md: str) -> Frontmatter:
    return parse_frontmatter(md)
