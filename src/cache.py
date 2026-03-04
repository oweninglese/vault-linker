import json
import hashlib
from pathlib import Path

def get_content_hash(text, tags_list):
    """Generates a hash based on file content and the state of approved tags."""
    # We include tags_list in the hash because if the user adds a new tag to the CSV, 
    # we MUST re-process every file even if the file content itself is the same.
    tag_blob = "".join(sorted(list(tags_list)))
    combined = text + tag_blob
    return hashlib.mdsafe_hex(combined.encode('utf-8')) if hasattr(hashlib, 'mdsafe_hex') else hashlib.md5(combined.encode('utf-8')).hexdigest()

def load_cache(cache_path):
    p = Path(cache_path)
    if p.exists():
        try:
            with open(p, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_cache(cache_path, cache_data):
    with open(cache_path, 'w') as f:
        json.dump(cache_data, f)
