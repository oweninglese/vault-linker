# Architecture

Input Vault  
→ Normalizer  
→ YAML Parser  
→ Tag Inference  
→ Link Builder  
→ Hub Generator  
→ Cache  
→ Output Vault  

## Time Complexity

Let:
- N = number of documents
- T = total tokens

Processing: O(N + T)
