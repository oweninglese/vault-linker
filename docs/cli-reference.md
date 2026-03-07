# CLI Reference

Vault-Linker currently exposes two commands:

run  
unlink

---

# run

Processes the vault normally.

Basic usage:

python -m src.cli run <vault> --tagfile TAGS.csv

## Important flags

--dry-run

Runs without writing changes.

--verbose

Shows progress.

--reindex

Forces a full rebuild.

--hub-dir

Sets the hub output directory.

--discover

Infers candidate tags.

--discover-min-count

Minimum frequency for candidate output.

--discover-acronyms

Includes acronym discovery.

--repair-frontmatter

Allows rewriting malformed frontmatter.

--json-report

Outputs the final report as JSON.

--linkify-mode

1 = once per file  
2 = once per paragraph (future)  
3 = all occurrences (future)

Current implementation supports only mode 1.

---

# unlink

Removes wikilinks only when their target is in the approved tagfile.

Basic usage:

python -m src.cli unlink <vault> --tagfile TAGS.csv

This is designed for controlled reset testing.

It allows you to:

1. strip approved-term links
2. rerun the linker
3. prove the engine is actually reinserting them

## Important flags

--dry-run

Preview removals without writing.

--verbose

Show progress.

--reindex

Force full pass.

--json-report

Output unlink results as JSON.

---

# Example reset flow

python -m src.cli unlink ~/vaults/vault --tagfile ~/vaults/tags/TAGS.csv

python -m src.cli run ~/vaults/vault --tagfile ~/vaults/tags/TAGS.csv --verbose
