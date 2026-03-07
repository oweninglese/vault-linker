# Troubleshooting

## No tags inserted

Check:

tagfile exists
tagfile contains terms
terms match text

---

## Files skipped

Vault-Linker skips files that are unchanged according to the index.

Use:

--reindex

to force a full rebuild.

---

## Encoding errors

Ensure files are valid UTF-8 or CP1252.

Invalid encodings are skipped.

---

## YAML parsing issues

Malformed YAML frontmatter is preserved rather than rewritten.

Fix manually if needed.
