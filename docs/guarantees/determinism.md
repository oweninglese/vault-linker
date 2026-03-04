# Deterministic Guarantees

Determinism means identical inputs produce identical:

- Index
- Plan
- File outputs
- Reports

## Requirements

- Stable file ordering (sorted paths)
- Stable tag ordering
- Stable link ordering
- No time-based metadata
- No random UUID generation

## Plan Stability

Plan must serialize identically for identical inputs.

Sorting hierarchy:

1. File path
2. Paragraph index
3. Match start index
