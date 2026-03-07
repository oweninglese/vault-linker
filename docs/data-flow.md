# Data Flow

This document explains how Vault-Linker processes a vault.

---

# Step 1: File Discovery

The scanner walks the vault directory.

Markdown files are discovered using filesystem traversal.

---

# Step 2: Cache Check

Each file is compared against the SQLite index using:

mtime
size
content hash

Unchanged files are skipped.

---

# Step 3: Markdown Parsing

Frontmatter is extracted.

Body content is separated.

---

# Step 4: Vocabulary Matching

Approved tags are compiled into an Aho-Corasick automaton.

The file body is scanned once.

Matches are returned as:

(start, end, term)

---

# Step 5: Link Insertion

The linker inserts wiki-links according to the configured linking rule.

Current behavior:

first occurrence per file.

---

# Step 6: Metadata Update

Metadata fields are merged:

title
author
source
tags
created

---

# Step 7: Index Update

The SQLite index records:

file metadata
term mentions

---

# Step 8: Hub Generation

Backlink hub pages are updated.

These pages list all notes referencing a tag.
