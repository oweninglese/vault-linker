#!/usr/bin/env bash
set -euo pipefail

ROOT="vault-linker"

mkdir -p \
  "$ROOT/.github/workflows" \
  "$ROOT/docs/DECISIONS" \
  "$ROOT/src/vault_linker/pipeline" \
  "$ROOT/src/vault_linker/stages" \
  "$ROOT/src/vault_linker/core" \
  "$ROOT/src/vault_linker/io" \
  "$ROOT/scripts" \
  "$ROOT/examples/example_vault" \
  "$ROOT/tests/unit" \
  "$ROOT/tests/integration" \
  "$ROOT/tests/fixtures" \
  "$ROOT/tests/golden" \
  "$ROOT/benchmarks/results" \
  "$ROOT/benchmarks/baselines"

# -------------------------
# ROOT FILES
# -------------------------

cat > "$ROOT/README.md" <<'MD'
# Vault-Linker

Structured Markdown Knowledge Graph Engine.

## Quick Start

~~~bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
vault-linker build ./examples/example_vault --output ./out_vault
~~~

See `docs/` for full documentation.
MD

cat > "$ROOT/pyproject.toml" <<'TOML'
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "vault-linker"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = ["PyYAML>=6.0"]

[project.scripts]
vault-linker = "vault_linker.cli:main"
TOML

cat > "$ROOT/.gitignore" <<'GIT'
__pycache__/
*.py[cod]
.venv/
dist/
build/
out_vault/
.cache/
benchmarks/results/*.json
GIT

# -------------------------
# DOCUMENTATION (SAFE FENCES)
# -------------------------

cat > "$ROOT/docs/ARCHITECTURE.md" <<'MD'
# Architecture

Input Vault  
→ Normalizer  
→ YAML Parser  
→ Tag Inference  
→ Link Builder  
→ Hub Generator  
→ Cache  
→ Output Vault  

## Time Complexity

Let:
- N = number of documents
- T = total tokens

Processing: O(N + T)
MD

cat > "$ROOT/docs/INSTALLATION.md" <<'MD'
# Installation

## From Zero

~~~bash
git clone <repo>
cd vault-linker
python -m venv .venv
source .venv/bin/activate
pip install -e .
~~~

## Run

~~~bash
vault-linker build ./examples/example_vault --output ./out_vault
~~~
MD

cat > "$ROOT/docs/MINIMAL_WORKING_EXAMPLE.md" <<'MD'
# Minimal Working Example

## Input

~~~markdown
---
title: Canada and Public Health
tags: [Canada]
---

Canada implemented new Public Health policy.
~~~

## Command

~~~bash
vault-linker build ./examples/example_vault --output ./out_vault
~~~
MD

cat > "$ROOT/docs/CONSTRAINTS_AND_FAILURES.md" <<'MD'
# Constraints & Failure Modes

## Assumptions
- UTF-8 files
- Valid YAML
- Markdown paragraph separation

## Failure Cases
- Malformed YAML → skip + log
- Tag case collision → canonicalize
- Cache corruption → rebuild
- Encoding error → reject
MD

cat > "$ROOT/docs/TESTING.md" <<'MD'
# Testing

## Strategy
- Unit tests for core logic
- Integration tests with fixture vaults
- Golden snapshot comparison

## Idempotence Rule

~~~text
build(build(vault)) == build(vault)
~~~
MD

cat > "$ROOT/docs/BENCHMARKS.md" <<'MD'
# Benchmarks

## Scenarios
1. Cold build
2. Warm build
3. Single file change
4. Worst-case linking

## Measurements
- total runtime
- stage runtime
- cache hit rate
MD

cat > "$ROOT/docs/DECISIONS/ADR-0001-determinism.md" <<'MD'
# ADR-0001: Determinism

Same input must produce identical output.
MD

cat > "$ROOT/docs/DECISIONS/ADR-0002-once-per-paragraph.md" <<'MD'
# ADR-0002: Once-Per-Paragraph Linking

At most one auto-link per paragraph.
MD

# -------------------------
# CLI SCAFFOLD
# -------------------------

cat > "$ROOT/src/vault_linker/__init__.py" <<'PY'
__version__ = "0.1.0"
PY

cat > "$ROOT/src/vault_linker/__main__.py" <<'PY'
from .cli import main
raise SystemExit(main())
PY

cat > "$ROOT/src/vault_linker/cli.py" <<'PY'
import argparse
from pathlib import Path

def build_cmd(args):
    in_vault = Path(args.vault)
    out_vault = Path(args.output)
    out_vault.mkdir(parents=True, exist_ok=True)
    (out_vault / "README_OUTPUT.md").write_text(
        f"Built from {in_vault}\n",
        encoding="utf-8",
    )
    print("Scaffold build complete.")
    return 0

def main():
    p = argparse.ArgumentParser(prog="vault-linker")
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("vault")
    b.add_argument("--output", required=True)
    b.set_defaults(func=build_cmd)
    args = p.parse_args()
    return args.func(args)
PY

# -------------------------
# BENCH SCRIPT
# -------------------------

cat > "$ROOT/scripts/benchmark.py" <<'PY'
import time
import json
from pathlib import Path

start = time.perf_counter()
time.sleep(0.01)
end = time.perf_counter()

out = {
    "total_seconds": end - start
}

Path("benchmarks/results").mkdir(parents=True, exist_ok=True)
Path("benchmarks/results/run.json").write_text(
    json.dumps(out, indent=2),
    encoding="utf-8",
)
print("Benchmark written.")
PY

# -------------------------
# EXAMPLE VAULT
# -------------------------

cat > "$ROOT/examples/example_vault/example.md" <<'MD'
---
title: Canada and Public Health
tags: [Canada]
---

Canada implemented new Public Health policy.
MD

echo "Scaffold complete."
