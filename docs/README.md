# Vault-Linker Documentation

Vault-Linker is a deterministic Markdown knowledge-graph linker.

It converts a plain Markdown vault into a structured system of:

- controlled vocabulary links
- backlink hub pages
- normalized metadata
- deterministic tagging

The system is designed to operate on large Markdown collections such as:

- Obsidian vaults
- research notebooks
- documentation repositories
- knowledge bases

---

# Documentation Index

## Getting Started

- [Quickstart](quickstart.md)
- [CLI Reference](cli-reference.md)

## System Design

- [Architecture](architecture.md)
- [Data Flow](data-flow.md)
- [Linking Algorithm](linking-algorithm.md)
- [SQLite Index](sqlite-index.md)

## File Formats

- [Tagfile Format](tagfile-format.md)
- [Hub Pages](hub-pages.md)

## Development

- [Testing](testing.md)
- [Troubleshooting](troubleshooting.md)
- [Known Limitations](known-limitations.md)
- [Roadmap](roadmap.md)

## Guarantees

Vault-Linker enforces several core guarantees:

- Deterministic behavior
- Idempotent runs
- Safe YAML handling
- Controlled linking constraints
- Encoding safety

These are documented in:

docs/guarantees/

---

# Philosophy

Vault-Linker intentionally avoids heavy infrastructure.

It uses:

- Markdown files
- deterministic algorithms
- SQLite indexing

to produce a knowledge graph while preserving plain text portability.
