# Quickstart

This guide shows how to run Vault-Linker on a Markdown vault.

---

# 1 Install

Clone the repository:

git clone https://github.com/oweninglese/vault-linker.git

Enter directory:

cd vault-linker

Create virtual environment:

python -m venv .venv

Activate:

source .venv/bin/activate

Install dependencies:

pip install -r requirements.txt

---

# 2 Prepare a vault

Example structure:

vault/
  notes/
    article1.md
    article2.md
  tags/
    TAGS.csv

---

# 3 Create a tag vocabulary

Example TAGS.csv:

Canada
Treaty 9
Artificial Intelligence
Machine Learning

---

# 4 Run Vault-Linker

python -m src.cli run ~/vaults/vault --tagfile ~/vaults/tags/TAGS.csv

---

# 5 What happens

Vault-Linker will:

1. scan Markdown files
2. match approved vocabulary terms
3. insert wiki links
4. normalize metadata
5. update backlink hubs
6. update the SQLite index

---

# 6 Controlled reset test

If you want to prove the engine is doing work, first strip approved-term links:

python -m src.cli unlink ~/vaults/vault --tagfile ~/vaults/tags/TAGS.csv

Then rerun the linker:

python -m src.cli run ~/vaults/vault --tagfile ~/vaults/tags/TAGS.csv --verbose

This gives you a clean before/after cycle.

---

# 7 Re-running

Running the tool again on an unchanged vault should result in zero modifications.

That is the idempotency guarantee.
