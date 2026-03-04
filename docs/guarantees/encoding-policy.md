# Encoding Policy

Vault-Linker must never corrupt files.

## Decode Order

1. UTF-8
2. UTF-8 with BOM
3. Fallback (explicitly defined)

Encoding used must be recorded in Document model.

## Write Policy

- Write using original encoding
- Atomic write (temp file + replace)
- Preserve newline style

Encoding errors must emit diagnostics.
