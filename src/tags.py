import sys
from pathlib import Path

def load_approved_tags(path):
    print(f"DEBUG: Opening tags file: {path}")
    tags = set()
    p = Path(path).expanduser().resolve()
    
    if not p.exists():
        print(f"DEBUG: File does NOT exist at {p}")
        return tags
    
    try:
        # Read the entire file content
        content = p.read_text(encoding="utf-8-sig").replace('\r', '')
        
        # Split by commas AND by newlines to handle both formats
        # This replaces newlines with commas then splits by comma
        raw_list = content.replace('\n', ',').split(',')
        
        print(f"DEBUG: Found {len(raw_list)} raw segments.")
        
        for item in raw_list:
            # Clean whitespace and quotes from each individual word
            tag = item.strip().strip('"')
            
            # Skip empty segments or headers
            if tag and tag.lower() not in ['tag', 'approved', 'category']:
                tags.add(tag)
        
        print(f"DEBUG: Successfully parsed {len(tags)} unique tags.")
    except Exception as e:
        print(f"DEBUG: Error during tag parsing: {e}")
        
    return tags

def extract_candidate_tags(idx):
    return sorted(list({d.title.strip() for d in idx.documents.values()}))

def write_tags_file(path, tags):
    with open(path, "w", encoding="utf-8") as f:
        # Save as a clean one-per-line list for future-proofing
        for t in sorted(list(tags)):
            f.write(f"{t}\n")
