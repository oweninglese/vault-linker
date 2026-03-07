# Testing

Vault-Linker should satisfy three core properties.

---

# Determinism

Running the tool on the same vault twice produces identical results.

---

# Idempotency

Second run should produce:

0 file modifications

---

# Safety

Ensure the tool does not modify:

code blocks
existing wiki links
markdown links

---

# Suggested tests

Run:

pytest tests/

Integration tests should verify:

link insertion
metadata merge
hub generation
