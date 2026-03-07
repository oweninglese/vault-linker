# Roadmap

Planned improvements for future versions.

---

# Linking Modes

Configurable linking density is now supported:

1 = once per file  
2 = once per paragraph  
3 = all occurrences

Future work may include:

- note-level defaults
- term-level density controls
- allowlists / denylists for aggressive mode 3 linking

---

# Candidate Discovery Improvements

Discovery should continue improving in these areas:

- better mixed phrase handling such as `Treaty 9`
- stronger rejection of generic single words
- better suppression of shorter generic fragments
- better domain-specific phrase promotion
- optional allowlists / denylists for vocabulary tuning

---

# Controlled Reset Options

The unlink command now supports vocabulary-aware reset.

Future improvements may include:

- unlink only body links
- unlink while preserving aliased display text exactly
- unlink by hub scope
- unlink by note subset

---

# Graph Export

Export the knowledge graph to formats such as:

GraphML  
JSON  
Neo4j-compatible edge lists

---

# Visualization

Generate graph diagrams and summary reports from the vault.

---

# Plugin Integration

Possible integrations:

Obsidian  
Logseq  
VS Code

---

# Parallel Processing

Improve performance on extremely large vaults through controlled parallelism.

---

# Richer Reports

Potential additions:

JSON reports per run  
benchmark snapshots  
diagnostic aggregation  
change summaries
