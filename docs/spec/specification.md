# Vault-Linker 3.0 — Authoritative Specification

Version 3.0 defines Vault-Linker as a deterministic Markdown transformation engine with a strict separation between indexing, planning, and application.

---

## Core Invariants

1. Deterministic output
2. Idempotent application
3. Encoding-safe IO
4. YAML frontmatter safety
5. Stable ordering of all operations
6. Pure planning phase (no side effects)

---

## Processing Phases

### Phase 1 — Index (Read-Only)

Produces structured models:

- Document
  - path
  - inferred_title (filename precedence)
  - frontmatter_raw
  - frontmatter_parsed (safe)
  - body_text
  - encoding_used
  - stats

- TagCatalog
  - approved_tags (from tagsfile)
  - inferred_candidates
  - missing_tagsfile flag

- LinkGraph
  - existing_links
  - proposed_links (not yet applied)

- Diagnostics

---

### Phase 2 — Plan (Pure)

Input: Index  
Output: Plan

Plan contains:
- Link insertions (file, paragraph, position)
- YAML tag append operations (attribute: "tags")
- No filesystem writes

Plan must be:
- Deterministic
- Fully ordered
- Reproducible

---

### Phase 3 — Apply (Side Effects)

Applies Plan:

- Atomic writes
- Context guards enforced
- No duplicate tag insertion
- No duplicate links
- No rewriting unchanged files

Produces:
- ApplyResult
- Structured report
- Diagnostics summary

---

## Title Inference Rule

Title precedence:

1. YAML "title" field (if valid)
2. Filename (without extension) — authoritative fallback

Filename inference is required for determinism.

---

## Tag Handling

Tags are appended only to YAML attribute named:

    tags

Rules:
- Must be a list
- Append only if not present
- Stable sorted order
- Respect --allow-missing-tagsfile

---

## Link Insertion Model

- Operates once per paragraph
- Context-aware
- Avoids:
  - Existing markdown links
  - Code blocks
  - Inline code
  - YAML frontmatter
- Enforces word boundary rules
- Stable matching order

---

## Required CLI Surface

- scan
- plan
- apply

Required flag:
- --allow-missing-tagsfile

---

Vault-Linker 3.0 is defined by guarantees, not features.
