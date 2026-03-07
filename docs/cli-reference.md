# CLI Reference

Vault-Linker currently exposes one main command:

vault-linker run

---

# Basic Usage

python -m src.cli run <vault> --tagfile TAGS.csv

---

# Required Arguments

vault

Path to the Markdown vault directory.

--tagfile

Path to the approved tag vocabulary file.

---

# Optional Flags

--dry-run

Runs the engine without writing changes.

---

--verbose

Prints progress information during execution.

---

--reindex

Forces a full rebuild of the SQLite cache/index.

---

--hub-dir

Directory where hub pages should be written.

Default:

vault root

---

--discover

Infers candidate tags from titles and note bodies.

---

--discover-min-count

Minimum vault-wide count required for a discovered candidate.

Default: 3

---

--discover-acronyms

Includes acronym detection during candidate discovery.

---

--repair-frontmatter

Allows rewriting malformed frontmatter into normalized YAML.

Use carefully.

---

--json-report

Outputs the final run report as JSON.

Useful for:

CI
automation
benchmark capture
future dashboards

---

--linkify-mode

Configures link insertion density.

1 = once per file  
2 = once per paragraph (future)  
3 = all occurrences (future)

Current implementation supports only:

1

Modes 2 and 3 are reserved and intentionally fail clearly.

---

# Example

python -m src.cli run ~/vaults/vault --tagfile ~/vaults/tags/TAGS.csv --verbose

---

# Example with JSON report

python -m src.cli run ~/vaults/vault \
  --tagfile ~/vaults/tags/TAGS.csv \
  --json-report

---

# Example future-facing linkify scaffold

python -m src.cli run ~/vaults/vault \
  --tagfile ~/vaults/tags/TAGS.csv \
  --linkify-mode 1
