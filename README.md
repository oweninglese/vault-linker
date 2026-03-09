
# Vault-Linker

**Vault-Linker** is a deterministic Markdown knowledge-linking engine that converts an unstructured Markdown vault into a structured, navigable knowledge graph.

It automatically:

- inserts approved wiki-links into notes
- normalizes YAML metadata
- generates backlink hub pages
- builds an incremental index for fast re-runs

Vault-Linker is designed for **large Markdown knowledge bases** such as Obsidian vaults, research notebooks, documentation systems, and personal knowledge archives.

---

# Table of Contents

- Overview
- Example: Before / After
- Architecture
- Processing Pipeline
- Algorithms Used
- Complexity Analysis
- Performance Characteristics
- System Design Rationale
- Safety Guarantees
- Expected Inputs
- Expected Outputs
- Installation
- Usage
- Configuration
- Benchmark Results
- Limitations
- Future Work

---

# Overview

Large knowledge bases accumulate information faster than links can be curated manually. This results in:

- fragmented knowledge
- missing cross-references
- inconsistent metadata
- difficult navigation

Vault-Linker solves this by enforcing **deterministic automated linking** based on an approved vocabulary.

The result is a Markdown vault that behaves like a **knowledge graph while remaining plain text.**

                INPUT SOURCES
         ┌────────────────────────┐
         │ PDFs • Markdown • Text │
         └────────────┬───────────┘
                      │
                      ▼
             OCR / TEXT EXTRACTION
         ┌─────────────────────────┐
         │ pdfminer / tesseract    │
         │ recover raw text        │
         └────────────┬────────────┘
                      │
                      ▼
            TEXT NORMALIZATION
         ┌─────────────────────────┐
         │ whitespace cleanup      │
         │ punctuation repair      │
         │ paragraph reconstruction│
         └────────────┬────────────┘
                      │
                      ▼
          DOCUMENT / VAULT NORMALIZATION
         ┌─────────────────────────┐
         │ frontmatter validation  │
         │ tag normalization       │
         │ filename policy         │
         │ link consistency        │
         └────────────┬────────────┘
                      │
                      ▼
           SEMANTIC TAGGING LAYER
         ┌─────────────────────────┐
         │ entity extraction       │
         │ vocabulary resolution   │
         │ flavor inference        │
         └────────────┬────────────┘
                      │
                      ▼
             KNOWLEDGE GRAPH
         ┌─────────────────────────┐
         │ cigars                  │
         │ brands                  │
         │ factories               │
         │ flavors                 │
         └────────────┬────────────┘
                      │
                      ▼
                WEBSITE / API
         ┌─────────────────────────┐
         │ search                  │
         │ graph exploration       │
         │ recommendations         │
         └─────────────────────────┘
Vault-Linker processes raw documents through a deterministic pipeline
consisting of extraction, text normalization, vault normalization,
semantic tagging, and graph construction.

Each stage is idempotent and produces structured artifacts that can be
re-run safely without corrupting the vault.
---

# Example: Before / After

### Original Note

Artificial intelligence has accelerated progress in machine learning
and neural networks over the past decade.

### After Vault-Linker

[[Artificial Intelligence]] has accelerated progress in
[[Machine Learning]] and [[Neural Networks]] over the past decade.

Each approved concept is linked **once per file** to avoid link clutter.

---

# Architecture

Vault-Linker is structured as a deterministic processing pipeline with modular components.

CLI → Configuration → Vocabulary → File Discovery → Cache → Parser → Matcher → Linker → Metadata → Atomic Write → Mention Index → Hub Generation

Each stage performs a single well‑defined responsibility.

---

# Processing Pipeline

1. CLI execution
2. Configuration loading
3. Vocabulary loading
4. Vault file discovery
5. SQLite cache check
6. Strict file decode
7. Markdown parsing
8. Multi-pattern matching
9. Metadata merge
10. Link insertion
11. Atomic file write
12. Mention index update
13. Hub page generation

---

# Algorithms Used

## Aho‑Corasick Pattern Matching

Vault-Linker uses the **Aho‑Corasick algorithm** to match vocabulary terms inside Markdown notes.

Definition:

Aho‑Corasick is a multi-pattern string matching algorithm that constructs a finite automaton capable of searching for many patterns simultaneously.

Advantages:

- deterministic behavior
- linear scan time
- excellent scaling with large vocabularies

---

# Complexity Analysis

Let:

n = characters in a note  
k = number of vocabulary terms  
m = total characters in vocabulary

Vocabulary compilation: O(m)

Note scan: O(n)

Total vault processing: O(total_text_size)

---

# Performance Characteristics

Vault-Linker performance depends primarily on:

- disk throughput
- vault size
- vocabulary size

Example performance:

| Vault Size | Terms | Runtime |
|------------|------|--------|
| 1k notes | 500 | ~1–2 seconds |
| 5k notes | 1000 | ~5–10 seconds |
| 20k notes | 2000 | ~30–60 seconds |

Subsequent runs are significantly faster due to caching.

---

# Incremental Processing

Vault-Linker maintains an SQLite index containing:

files  
mentions

The index stores:

- file path
- modification time
- file size
- content hash
- approved term mentions

If a file has not changed, it is skipped entirely.

---

# System Design Rationale

### SQLite vs Flat Cache

SQLite was chosen because it provides:

- structured data storage
- scalable indexing
- fast incremental updates
- queryable backlinks

### Deterministic Processing

Vault-Linker processes files in sorted order and normalizes metadata so identical inputs produce identical outputs.

### Atomic Writes

Files are written safely using:

1. write temporary file
2. replace original file

### YAML Preservation

Malformed frontmatter is preserved rather than aggressively repaired to protect user data.

---

# Safety Guarantees

Vault-Linker avoids modifying:

- fenced code blocks
- inline code
- existing wiki-links
- Markdown links
- URLs

---

# Expected Inputs

Vault-Linker expects:

- a directory containing Markdown files
- a vocabulary file defining approved concepts
- text files decodable under configured encodings

---

# Expected Outputs

After execution Vault-Linker may:

- insert wiki-links
- normalize YAML frontmatter
- merge tags
- update the SQLite index
- generate hub pages

---

# Installation

Clone repository:

git clone https://github.com/yourusername/vault-linker.git

Enter project:

cd vault-linker

Create environment:

python -m venv .venv

Activate:

source .venv/bin/activate

Install dependencies:

pip install -r requirements.txt

---

# Usage

Run Vault-Linker:

python -m src.cli run /path/to/vault --tagfile /path/to/tags.txt

Example:

python -m src.cli run ~/vaults/vault --tagfile ~/vaults/tags/TAGS.csv

---

# Useful Options

Dry run:

--dry-run

Verbose:

--verbose

Force reindex:

--reindex

Candidate discovery:

--discover

---

# Benchmark Example

Vault size: 1274 notes  
Vocabulary: 943 terms  
Runtime: ~3 seconds  
Links inserted: ~9000

Subsequent runs typically process almost zero files due to caching.

---

# Limitations

- linking is lexical rather than semantic
- each term linked once per file
- malformed YAML preserved rather than repaired

---

# Future Work

Possible improvements:

- semantic entity detection
- block-level linking
- graph visualizations
- Obsidian plugin integration
- tag suggestion interface

---

# Author

**Owen Inglese**

Vault-Linker explores deterministic knowledge organization, automated linking systems, and scalable Markdown knowledge graphs.
