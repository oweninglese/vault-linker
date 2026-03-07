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

# Current Linking Policy

Current policy:

once per file

That means the first valid occurrence of each term is linked, and later occurrences in the same note are left alone.

This was chosen because it:

- avoids link spam
- keeps notes readable
- produces stable diffs
- reduces accidental overlinking

---

# Protected Contexts

Vault-Linker does not insert links inside:

- fenced code blocks
- inline code
- existing wiki-links
- Markdown links
- raw URLs

---

# Future Feature: Linkify Ratio

A configurable linking density is planned.

Option | Behavior
------ | --------
1 | once per file
2 | once per paragraph
3 | all valid occurrences

This will likely be exposed as:

--linkify-mode

Current status:

mode 1 implemented  
modes 2 and 3 reserved for future work
