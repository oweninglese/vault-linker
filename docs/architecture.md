# Architecture

Vault-Linker is structured as a deterministic pipeline.

---

# System Overview

Input Vault
↓
File Scanner
↓
Metadata Parser
↓
Vocabulary Matcher
↓
Link Transformer
↓
Metadata Merger
↓
SQLite Index
↓
Hub Generator
↓
Updated Vault

---

# Core Components

Scanner

Discovers Markdown files and builds a document index.

Parser

Extracts YAML frontmatter and body content.

Vocabulary Engine

Loads approved tag vocabulary.

Matcher

Uses Aho-Corasick multi-pattern search.

Linker

Applies linking rules to the first occurrence of each term.

Metadata Engine

Infers:

title
author
source
tags

Index

Stores file metadata and mention relationships in SQLite.

Hub Generator

Creates backlink hub pages.

---

# Key Design Goals

Deterministic behavior

Same inputs always produce identical outputs.

Idempotency

Running twice produces zero modifications.

Safety

Avoids modifying:

code blocks
existing links
YAML content
