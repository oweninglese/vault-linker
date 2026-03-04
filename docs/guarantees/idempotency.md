# Idempotency Rules

Idempotency means:

Running `apply` twice results in no further changes.

## Link Rules

- If link already exists, do not reinsert
- If text already wrapped in link, skip

## YAML Rules

- If tag already present in `tags`, do not append
- Do not reorder existing tags unless normalization required
- Never duplicate tags

## File Writes

Only write file if content changes.
