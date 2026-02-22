
#!/usr/bin/env bash
set -euo pipefail

# If we're already inside the repo (src/vault_linker exists), use "."
# Otherwise, fall back to "./vault-linker"
if [[ -d "src/vault_linker" ]]; then
  ROOT="."
elif [[ -d "vault-linker/src/vault_linker" ]]; then
  ROOT="vault-linker"
else
  echo "ERROR: cannot find repo. Expected either:"
  echo "  ./src/vault_linker (run from inside repo), or"
  echo "  ./vault-linker/src/vault_linker (run from parent dir)."
  exit 1
fi

# -----------------------------
# src/vault_linker/logging.py
# -----------------------------
cat > "$ROOT/src/vault_linker/logging.py" <<'PY'
from __future__ import annotations

import logging
import sys

def get_logger(name: str = "vault_linker") -> logging.Logger:
    log = logging.getLogger(name)
    if log.handlers:
        return log
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    log.addHandler(h)
    log.setLevel(logging.INFO)
    return log
PY

# -----------------------------
# src/vault_linker/config.py
# -----------------------------
cat > "$ROOT/src/vault_linker/config.py" <<'PY'
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
    learn_tags: bool = False
    dry_run: bool = False
PY

# -----------------------------
# src/vault_linker/io/fs.py
# -----------------------------
cat > "$ROOT/src/vault_linker/io/fs.py" <<'PY'
from __future__ import annotations

from pathlib import Path
from typing import Iterable

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def iter_md_files(root: Path) -> Iterable[Path]:
    for p in sorted(root.rglob("*.md")):
        if p.is_file():
            yield p

def relpath_under(root: Path, p: Path) -> Path:
    return p.resolve().relative_to(root.resolve())

def read_text_utf8(p: Path) -> str:
    data = p.read_bytes()
    return data.decode("utf-8")

def write_text_utf8(p: Path, s: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(s, encoding="utf-8", newline="\n")
PY

# -----------------------------
# src/vault_linker/core/hashing.py
# -----------------------------
cat > "$ROOT/src/vault_linker/core/hashing.py" <<'PY'
from __future__ import annotations

import hashlib

def sha256_text(s: str) -> str:
    h = hashlib.sha256()
    h.update(s.encode("utf-8"))
    return h.hexdigest()
PY

# -----------------------------
# src/vault_linker/core/yaml.py
# -----------------------------
cat > "$ROOT/src/vault_linker/core/yaml.py" <<'PY'
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

def dump_frontmatter(data: Dict[str, Any]) -> str:
    y = yaml.safe_dump(
        data,
        sort_keys=True,
        allow_unicode=True,
        default_flow_style=False,
        width=88,
    ).strip()
    return f"---\n{y}\n---\n\n"
PY

# -----------------------------
# src/vault_linker/core/tags.py
# -----------------------------
cat > "$ROOT/src/vault_linker/core/tags.py" <<'PY'
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, List, Set

STOPWORDS = {
    "a","an","the","and","or","of","to","in","on","for","with","by","from",
    "is","are","was","were","be","been","being",
}

def _clean_token(t: str) -> str:
    t = t.strip()
    t = re.sub(r"[^\w\- ]+", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def canonicalize_tag(t: str) -> str:
    t = _clean_token(t)
    t = t.replace(" ", "-")
    t = re.sub(r"-{2,}", "-", t)
    return t

def normalize_tag_list(v) -> List[str]:
    if v is None:
        return []
    if isinstance(v, list):
        raw = [str(x) for x in v]
    elif isinstance(v, str):
        raw = [x.strip() for x in v.split(",") if x.strip()]
    else:
        raw = [str(v)]
    out: List[str] = []
    seen = set()
    for t in raw:
        c = canonicalize_tag(t)
        if not c:
            continue
        k = c.casefold()
        if k not in seen:
            out.append(c)
            seen.add(k)
    return out

def read_tags_file(p: Path) -> List[str]:
    if not p.exists():
        return []
    lines = p.read_text(encoding="utf-8").splitlines()
    tags: List[str] = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        c = canonicalize_tag(line)
        if c:
            tags.append(c)
    out: List[str] = []
    seen = set()
    for t in tags:
        k = t.casefold()
        if k not in seen:
            out.append(t)
            seen.add(k)
    return out

def append_tags_file(p: Path, new_tags: Iterable[str]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    existing = read_tags_file(p)
    existing_ci: Set[str] = {t.casefold() for t in existing}
    add: List[str] = []
    for t in new_tags:
        c = canonicalize_tag(t)
        if c and c.casefold() not in existing_ci:
            add.append(c)
            existing_ci.add(c.casefold())
    if not add:
        return
    add_sorted = sorted(add, key=lambda x: x.casefold())
    with p.open("a", encoding="utf-8", newline="\n") as f:
        for t in add_sorted:
            f.write(t + "\n")

def infer_from_title(title: str) -> List[str]:
    toks = re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", title or "")
    out = []
    for t in toks:
        if t.casefold() in STOPWORDS:
            continue
        out.append(canonicalize_tag(t))
    return _uniq_ci(out)

def infer_from_body(body: str) -> List[str]:
    toks = re.findall(r"\b[A-Z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)*\b", body or "")
    out = []
    for t in toks:
        if len(t) < 3:
            continue
        if t.casefold() in STOPWORDS:
            continue
        out.append(canonicalize_tag(t))
    return _uniq_ci(out)

def _uniq_ci(xs: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for x in xs:
        if not x:
            continue
        k = x.casefold()
        if k not in seen:
            out.append(x)
            seen.add(k)
    return out
PY

# -----------------------------
# src/vault_linker/core/markdown.py
# -----------------------------
cat > "$ROOT/src/vault_linker/core/markdown.py" <<'PY'
from __future__ import annotations

import re
from typing import List

def normalize_md(md: str) -> str:
    md = md.replace("\r\n", "\n").replace("\r", "\n")
    lines = [ln.rstrip() for ln in md.split("\n")]
    md = "\n".join(lines)
    md = re.sub(r"\n{3,}", "\n\n", md)
    if not md.endswith("\n"):
        md += "\n"
    return md

def split_paragraphs(body: str) -> List[str]:
    body = body.replace("\r\n", "\n").replace("\r", "\n")
    parts = re.split(r"\n\s*\n", body.strip("\n"))
    return [p.strip("\n") for p in parts if p.strip() != ""]

def join_paragraphs(paragraphs: List[str]) -> str:
    return "\n\n".join(p.rstrip() for p in paragraphs).rstrip() + "\n"
PY

# -----------------------------
# src/vault_linker/core/links.py
# -----------------------------
cat > "$ROOT/src/vault_linker/core/links.py" <<'PY'
from __future__ import annotations

import re
from typing import Dict, List, Tuple

def _is_already_linked(paragraph: str, start: int, end: int) -> bool:
    # wiki link guard
    window = paragraph[max(0, start - 2) : min(len(paragraph), end + 2)]
    if "[[" in window and "]]" in window:
        return True
    # markdown link guard (crude but safe)
    if start > 0 and paragraph[start - 1] in "[(":
        return True
    if end < len(paragraph) and paragraph[end : end + 1] in "]":
        return True
    return False

def link_once_per_paragraph(
    paragraph: str,
    targets: List[str],
    *,
    wiki: bool = True,
) -> Tuple[str, Dict[str, int]]:
    counts: Dict[str, int] = {}
    out = paragraph

    for t in targets:
        pat = re.compile(rf"(?<![\w\-]){re.escape(t)}(?![\w\-])")
        m = pat.search(out)
        if not m:
            continue
        if _is_already_linked(out, m.start(), m.end()):
            continue
        repl = f"[[{t}]]" if wiki else f"[{t}]({t}.md)"
        out = out[: m.start()] + repl + out[m.end() :]
        counts[t] = 1
    return out, counts
PY

# -----------------------------
# src/vault_linker/stages/cache.py
# -----------------------------
cat > "$ROOT/src/vault_linker/stages/cache.py" <<'PY'
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

def should_process(relpath: str, normalized_input: str, cache: CacheState) -> bool:
    h = sha256_text(normalized_input)
    return cache.files.get(relpath) != h

def update_cache(relpath: str, normalized_input: str, cache: CacheState) -> None:
    cache.files[relpath] = sha256_text(normalized_input)
PY

# -----------------------------
# src/vault_linker/stages/*
# -----------------------------
cat > "$ROOT/src/vault_linker/stages/ingest.py" <<'PY'
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
PY

cat > "$ROOT/src/vault_linker/stages/normalize.py" <<'PY'
from __future__ import annotations
from vault_linker.core.markdown import normalize_md
def normalize(md: str) -> str:
    return normalize_md(md)
PY

cat > "$ROOT/src/vault_linker/stages/yaml_parse.py" <<'PY'
from __future__ import annotations
from vault_linker.core.yaml import Frontmatter, parse_frontmatter
def parse(md: str) -> Frontmatter:
    return parse_frontmatter(md)
PY

cat > "$ROOT/src/vault_linker/stages/infer_tags.py" <<'PY'
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Set

from vault_linker.core.tags import infer_from_body, infer_from_title, normalize_tag_list

@dataclass(frozen=True)
class TagResult:
    tags: List[str]
    new_candidates: List[str]

def infer_tags(
    *,
    title: str,
    body: str,
    existing_registry: List[str],
    yaml_tags,
    learn_tags: bool,
) -> TagResult:
    reg_ci: Set[str] = {t.casefold() for t in existing_registry}
    base = normalize_tag_list(yaml_tags)
    candidates: List[str] = []
    candidates.extend(infer_from_title(title))
    candidates.extend(infer_from_body(body))

    cand_u: List[str] = []
    seen = set()
    for t in candidates:
        k = t.casefold()
        if k not in seen:
            cand_u.append(t)
            seen.add(k)

    new = [t for t in cand_u if t.casefold() not in reg_ci]

    if learn_tags:
        final = _merge_ci(base + existing_registry + cand_u)
    else:
        final = _merge_ci(base + existing_registry)

    return TagResult(tags=final, new_candidates=new)

def _merge_ci(xs: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for x in xs:
        if not x:
            continue
        k = x.casefold()
        if k not in seen:
            out.append(x)
            seen.add(k)
    return out
PY

cat > "$ROOT/src/vault_linker/stages/link_body.py" <<'PY'
from __future__ import annotations

from typing import List

from vault_linker.core.links import link_once_per_paragraph
from vault_linker.core.markdown import join_paragraphs, split_paragraphs

def link_body(body: str, tags_to_link: List[str]) -> str:
    tags_sorted = sorted(tags_to_link, key=lambda x: x.casefold())
    paras = split_paragraphs(body)
    out_paras = []
    for p in paras:
        new_p, _ = link_once_per_paragraph(p, tags_sorted, wiki=True)
        out_paras.append(new_p)
    return join_paragraphs(out_paras)
PY

cat > "$ROOT/src/vault_linker/stages/hubs.py" <<'PY'
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from vault_linker.io.fs import write_text_utf8

@dataclass
class HubIndex:
    refs: Dict[str, List[str]]

def build_hub_index() -> HubIndex:
    return HubIndex(refs={})

def add_refs(index: HubIndex, rel_doc: str, tags: List[str]) -> None:
    for t in tags:
        index.refs.setdefault(t, []).append(rel_doc)

def write_hubs(output_root: Path, index: HubIndex) -> None:
    for tag in sorted(index.refs.keys(), key=lambda x: x.casefold()):
        refs = sorted(set(index.refs[tag]), key=lambda x: x.casefold())
        lines = [f"# {tag}", "", "Referenced By:", ""]
        for r in refs:
            lines.append(f"- {r}")
        lines.append("")
        write_text_utf8(output_root / f"{tag}.md", "\n".join(lines))
PY

# -----------------------------
# src/vault_linker/pipeline/build.py
# -----------------------------
cat > "$ROOT/src/vault_linker/pipeline/build.py" <<'PY'
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

from vault_linker.config import BuildConfig
from vault_linker.logging import get_logger
from vault_linker.io.fs import ensure_dir, read_text_utf8, relpath_under, write_text_utf8
from vault_linker.core.tags import append_tags_file, normalize_tag_list, read_tags_file
from vault_linker.core.yaml import dump_frontmatter
from vault_linker.stages.cache import load_cache, save_cache, should_process, update_cache
from vault_linker.stages.ingest import ingest
from vault_linker.stages.normalize import normalize
from vault_linker.stages.yaml_parse import parse
from vault_linker.stages.infer_tags import infer_tags
from vault_linker.stages.link_body import link_body
from vault_linker.stages.hubs import add_refs, build_hub_index, write_hubs

log = get_logger()

@dataclass(frozen=True)
class BuildStats:
    total: int
    processed: int
    skipped: int
    yaml_errors: int
    learned_tags: int

def build(cfg: BuildConfig) -> BuildStats:
    ensure_dir(cfg.output_vault)
    ensure_dir(cfg.cache_dir)

    cache = load_cache(cfg.cache_dir)
    registry = read_tags_file(cfg.tags_file)
    hub_index = build_hub_index()

    items = ingest(cfg.input_vault)

    processed = 0
    skipped = 0
    yaml_errors = 0
    learned: List[str] = []

    for item in items:
        rel = relpath_under(cfg.input_vault, item.path)
        rel_str = str(rel.as_posix())

        try:
            raw = read_text_utf8(item.path)
        except UnicodeDecodeError:
            log.error(f"Non-UTF8 rejected: {rel_str}")
            skipped += 1
            continue

        normalized = normalize(raw)

        if not should_process(rel_str, normalized, cache):
            skipped += 1
            continue

        try:
            fm = parse(normalized)
        except ValueError as e:
            log.error(f"YAML error in {rel_str}: {e}")
            yaml_errors += 1
            skipped += 1
            continue

        title = str(fm.data.get("title", "") or "")
        yaml_tags = fm.data.get("tags", None)

        tag_res = infer_tags(
            title=title,
            body=fm.body,
            existing_registry=registry,
            yaml_tags=yaml_tags,
            learn_tags=cfg.learn_tags,
        )

        fm.data["tags"] = normalize_tag_list(tag_res.tags)

        if cfg.learn_tags and tag_res.new_candidates:
            learned.extend(tag_res.new_candidates)

        body_out = fm.body
        if not cfg.no_links:
            body_out = link_body(body_out, fm.data["tags"])

        hashtags = " ".join(f"#{t}" for t in fm.data["tags"])
        if hashtags:
            body_out = body_out.rstrip("\n") + "\n\n" + hashtags + "\n"

        out_text = dump_frontmatter(fm.data) + body_out.lstrip("\n")

        out_path = (cfg.output_vault / rel).resolve()
        if cfg.dry_run:
            log.info(f"[dry-run] would write: {rel_str}")
        else:
            write_text_utf8(out_path, out_text)
            update_cache(rel_str, normalized, cache)

        add_refs(hub_index, rel_str, fm.data["tags"])
        processed += 1

    if not cfg.dry_run:
        if cfg.learn_tags and learned:
            append_tags_file(cfg.tags_file, learned)
        write_hubs(cfg.output_vault, hub_index)
        save_cache(cfg.cache_dir, cache)

    learned_count = len({t.casefold(): t for t in learned})
    return BuildStats(
        total=len(items),
        processed=processed,
        skipped=skipped,
        yaml_errors=yaml_errors,
        learned_tags=learned_count,
    )
PY

# -----------------------------
# src/vault_linker/cli.py
# -----------------------------
cat > "$ROOT/src/vault_linker/cli.py" <<'PY'
from __future__ import annotations

import argparse
from pathlib import Path

from vault_linker.config import BuildConfig
from vault_linker.logging import get_logger
from vault_linker.pipeline.build import build

log = get_logger()

def _default_tags_file() -> Path:
    return Path.home() / "library" / "tags" / "TAGS.txt"

def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="vault-linker")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="Build an output vault from an input vault")
    b.add_argument("vault", help="Path to input vault")
    b.add_argument("--output", required=True, help="Path to output vault")
    b.add_argument("--cache-dir", default=None, help="Cache dir (default: <output>/.vault_linker_cache)")
    b.add_argument("--tags-file", default=None, help="Tag registry file (default: ~/library/tags/TAGS.txt)")
    b.add_argument("--no-links", action="store_true", help="Disable body linking")
    b.add_argument("--learn-tags", action="store_true", help="Learn inferred tags into tag registry")
    b.add_argument("--dry-run", action="store_true", help="No writes; report actions")

    args = p.parse_args(argv)

    input_vault = Path(args.vault).expanduser().resolve()
    output_vault = Path(args.output).expanduser().resolve()
    cache_dir = Path(args.cache_dir).expanduser().resolve() if args.cache_dir else (output_vault / ".vault_linker_cache")

    tags_file = Path(args.tags_file).expanduser().resolve() if args.tags_file else _default_tags_file()
    if args.tags_file is None and not tags_file.exists():
        tags_file = input_vault / "TAGS.txt"

    cfg = BuildConfig(
        input_vault=input_vault,
        output_vault=output_vault,
        cache_dir=cache_dir,
        tags_file=tags_file,
        no_links=bool(args.no_links),
        learn_tags=bool(args.learn_tags),
        dry_run=bool(args.dry_run),
    )

    stats = build(cfg)
    log.info(
        f"done: total={stats.total} processed={stats.processed} skipped={stats.skipped} "
        f"yaml_errors={stats.yaml_errors} learned_tags={stats.learned_tags}"
    )
    return 0
PY

# -----------------------------
# tests (real)
# -----------------------------
cat > "$ROOT/tests/unit/test_tags_normalization.py" <<'PY'
from vault_linker.core.tags import normalize_tag_list, canonicalize_tag

def test_canonicalize_tag() -> None:
    assert canonicalize_tag(" Public Health ") == "Public-Health"

def test_normalize_tag_list() -> None:
    assert normalize_tag_list(["Canada", "canada", "Public Health"]) == ["Canada", "Public-Health"]
PY

cat > "$ROOT/tests/integration/test_build_example_vault.py" <<'PY'
from pathlib import Path
import tempfile

from vault_linker.config import BuildConfig
from vault_linker.pipeline.build import build

def test_build_example_vault_creates_output_and_hubs() -> None:
    repo = Path(__file__).resolve().parents[2]
    input_vault = repo / "examples" / "example_vault"

    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "out"
        cache = Path(td) / "cache"
        tags = Path(td) / "TAGS.txt"

        cfg = BuildConfig(
            input_vault=input_vault,
            output_vault=out,
            cache_dir=cache,
            tags_file=tags,
            no_links=False,
            learn_tags=True,
            dry_run=False,
        )
        stats = build(cfg)

        assert stats.processed >= 1
        assert (out / "example.md").exists()
        assert (out / "Canada.md").exists()
PY

echo "OK: pipeline wired into existing repo at: $ROOT"
