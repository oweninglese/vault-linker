# Link Insertion Constraints

## Context Guards

Never insert inside:

- YAML frontmatter
- Code blocks (``` fenced)
- Inline code (`code`)
- Existing markdown links

## Paragraph Model

Insertion occurs once per paragraph pass.

## Matching Rules

- Whole word matches only
- Case sensitivity policy explicitly defined
- Stable first-match policy

Deterministic matching is mandatory.
