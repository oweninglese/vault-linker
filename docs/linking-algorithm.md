# Linking Algorithm

Vault-Linker uses the Aho-Corasick algorithm for multi-pattern matching.

---

# Why Aho-Corasick

A naive linker would scan the same file once for each approved term.

That scales poorly.

If:

n = note length  
k = number of terms

naive repeated matching behaves roughly like:

O(n * k)

Vault-Linker instead compiles the vocabulary into an automaton and scans the text once.

That gives approximately:

O(n)

for the note scan, after automaton construction.

---

# Linking Modes

Vault-Linker supports three link density modes.

## Mode 1

once per file

The first valid occurrence of each term is linked in the file body.

## Mode 2

once per paragraph

The first valid occurrence of each term is linked independently within each paragraph.

## Mode 3

all valid occurrences

Every valid plain-text occurrence is linked.

---

# Protected Contexts

All modes skip:

- fenced code blocks
- inline code
- existing wiki-links
- Markdown links
- raw URLs

---

# Idempotency

The linker treats existing wikilinks as already linked.

That prevents repeated runs from progressively linking the second, third, or later occurrences of the same term in mode 1 or mode 2.

Mode 3 is also idempotent because existing wikilinks are skipped as protected spans.

---

# Trade-off

Mode 1 is the least noisy and safest default.

Mode 2 is useful when long notes need local paragraph-level navigability.

Mode 3 is the most aggressive and can create dense link coverage, which may be desirable for certain research or graph-heavy vaults.
