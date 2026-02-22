from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import yaml

@dataclass(frozen=True)
class Frontmatter:
    data: Dict[str, Any]
    body: str

def split_frontmatter(md: str) -> Tuple[str, str]:
    if not md.startswith("---"):
        return "", md

    lines = md.splitlines(keepends=True)
    if not lines or lines[0].strip() != "---":
        return "", md

    end_idx = None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end_idx = i
            break
    if end_idx is None:
        raise ValueError("Malformed YAML frontmatter: missing closing '---'")

    yaml_text = "".join(lines[1:end_idx])
    body = "".join(lines[end_idx + 1 :])
    return yaml_text, body.lstrip("\n")

def parse_frontmatter(md: str) -> Frontmatter:
    yaml_text, body = split_frontmatter(md)
    if not yaml_text:
        return Frontmatter(data={}, body=body)
    try:
        data = yaml.safe_load(yaml_text) or {}
    except Exception as e:
        raise ValueError(f"Malformed YAML frontmatter: {e}") from e
    if not isinstance(data, dict):
        raise ValueError("Malformed YAML frontmatter: top-level must be a mapping")
    return Frontmatter(data=data, body=body)

def _nullish(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, str) and v.strip().lower() in {"null", "none", ""}:
        return True
    return False

def _clean_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Remove null placeholders, but ALWAYS keep 'tags' key (even if empty list).
    """
    out: Dict[str, Any] = {}
    for k, v in (data or {}).items():
        if k != "tags" and _nullish(v):
            continue
        out[k] = v

    # ensure tags key exists
    if "tags" not in out or out["tags"] is None:
        out["tags"] = []
    if isinstance(out["tags"], tuple):
        out["tags"] = list(out["tags"])
    if not isinstance(out["tags"], list):
        out["tags"] = [str(out["tags"])]

    return out

def dump_frontmatter(data: Dict[str, Any]) -> str:
    data = _clean_data(data)
    y = yaml.safe_dump(
        data,
        sort_keys=True,
        allow_unicode=True,
        default_flow_style=False,
        width=88,
    ).strip()
    return f"---\n{y}\n---\n\n"
