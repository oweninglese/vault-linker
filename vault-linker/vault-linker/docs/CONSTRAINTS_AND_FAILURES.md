# Constraints & Failure Modes

## Assumptions
- UTF-8 files
- Valid YAML
- Markdown paragraph separation

## Failure Cases
- Malformed YAML → skip + log
- Tag case collision → canonicalize
- Cache corruption → rebuild
- Encoding error → reject
