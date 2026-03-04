# YAML Safety Rules

Frontmatter must be parsed safely.

## Requirements

- Use safe loader only
- No object construction
- No execution of arbitrary YAML tags

## Schema Rules

- tags must be list or convertible to list
- title must be string
- Unknown fields preserved in raw frontmatter

If invalid YAML:
- Preserve raw frontmatter
- Emit diagnostic
- Do not crash
