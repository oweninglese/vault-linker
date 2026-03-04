
# vault-linker

Fast Obsidian vault linker + hub generator.

## What it does

- Scans a vault for `.md` files
- Extracts frontmatter `tags:` if present (fast YAML)
- Finds mentions of hub terms in the body (word-boundary safe)
- Inserts Obsidian-native wiki links `[[Term]]` once per paragraph (optional)
- Maintains hub pages (e.g. `Canada.md`) by regenerating only a bounded backlinks block

## Safety

- Only modifies text between marker comments:
  - <!-- vault-linker:hub:start -->
  - <!-- vault-linker:hub:end -->

## Install (editable)

python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .

## Run

vault-linker run /path/to/vault --verbose

Dry-run (no writes):
vault-linker run /path/to/vault --dry-run --verbose
