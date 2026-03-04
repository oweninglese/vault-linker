import re
import yaml
from datetime import datetime

def parse_markdown_content(text):
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    if not text.startswith('---'):
        return {}, None, text.strip(), []
    
    parts = text.split('---', 2)
    if len(parts) < 3:
        return {}, None, text.strip(), []

    raw_yaml = parts[1]
    body = parts[2].strip()
    
    try:
        data = yaml.safe_load(raw_yaml)
        # CRITICAL: Clean any quotes out of the keys themselves
        if isinstance(data, dict):
            clean_data = {}
            for k, v in data.items():
                clean_key = str(k).replace("'", "").replace('"', "").strip()
                clean_data[clean_key] = v
            data = clean_data
        return (data if isinstance(data, dict) else {}), raw_yaml, body, []
    except:
        return {}, raw_yaml, body, []

def enrich_metadata(metadata, path_obj, body):
    if not metadata.get('title'):
        metadata['title'] = path_obj.stem
    if not metadata.get('created'):
        stat = path_obj.stat()
        dt = datetime.fromtimestamp(stat.st_mtime)
        metadata['created'] = dt.strftime('%Y-%m-%d %H:%M')

    if not metadata.get('author') or "(s):" in str(metadata.get('author')):
        # Only look at the start of the body
        sample = "\n".join(body.splitlines()[:15])
        match = re.search(r'(?i)^\s*(?:author(?:\(s\))?|by)\s*:\s*(.*)', sample, re.MULTILINE)
        if match:
            metadata['author'] = match.group(1).strip()
    return metadata

def update_frontmatter_tags(metadata, discovered_tags):
    existing = metadata.get('tags', [])
    if isinstance(existing, str):
        existing = [t.strip() for t in existing.split(',') if t.strip()]
    elif not isinstance(existing, list):
        existing = []
    
    combined = sorted(list(set(str(t) for t in existing) | set(discovered_tags)))
    if combined:
        metadata['tags'] = combined
    return metadata

def stringify_frontmatter(metadata):
    if not metadata: return ""
    # NO default_style ensures keys are unquoted.
    # NO default_flow_style ensures the list looks like:
    # tags:
    # - tag1
    yaml_text = yaml.safe_dump(
        metadata, 
        sort_keys=True, 
        allow_unicode=True, 
        width=1000,
        default_flow_style=False
    ).strip()
    return f"---\n{yaml_text}\n---\n\n"
