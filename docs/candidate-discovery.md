# Candidate Discovery

Vault-Linker can infer possible tag candidates from note titles and bodies.

This is intended to help refine the approved vocabulary over time.

---

# What It Tries To Find

The discovery system prefers:

- multi-word Title Case phrases
- mixed concept phrases such as `Treaty 9`
- selected uppercase/alphanumeric concepts such as `G7` or `COVID-19`
- hyphenated domain terms
- selected proper nouns

Examples of desirable candidates:

- Treaty 9
- James Bay
- Moose Cree First Nation
- Machine Learning
- COVID-19

---

# What It Rejects

The discovery system tries to avoid low-value candidates such as:

- one
- two
- first
- second
- history
- north
- south
- region
- arbitrary lowercase word + number phrases such as `and 300` or `almost 40`

It is intentionally conservative with single-word candidates.

---

# Overlap Suppression

If a more specific phrase is found, shorter fragments may be suppressed.

Examples:

- keep `Treaty 9`, drop `Treaty`
- keep `James Bay`, drop `Bay`

---

# Current Limitations

Candidate discovery is lexical, not semantic.

It does not understand meaning or context beyond pattern matching and filtering.

That means:

- some valid concepts may still be missed
- some noisy candidates may still appear
- the output should be reviewed manually

---

# Example Command

python -m src.cli run ~/vaults/vault \
  --tagfile ~/vaults/tags/TAGS.csv \
  --hub-dir .vault-linker/hubs \
  --discover \
  --discover-min-count 1 \
  --discover-out ~/vaults/vault/.vault-linker/tag_candidates.txt \
  --reindex \
  --verbose
