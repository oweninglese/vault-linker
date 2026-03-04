import pathlib
import os
from frontmatter import parse_markdown_content

class Document:
    def __init__(self, path, frontmatter, body, encoding, fingerprint):
        self.path = path
        self.frontmatter = frontmatter
        self.body = body
        self.encoding = encoding
        self.fingerprint = fingerprint

class VaultIndex:
    def __init__(self):
        self.documents = {}

def scan(vault_path, config):
    index = VaultIndex()
    vault = pathlib.Path(vault_path)
    
    for md_file in vault.rglob("*.md"):
        stat = md_file.stat()
        # Fingerprint = "Size-LastModified"
        fingerprint = f"{stat.st_size}-{stat.st_mtime}"
        
        with open(md_file, 'rb') as f:
            raw_bytes = f.read()
            
        encoding = 'utf-8'
        try:
            content = raw_bytes.decode(encoding)
        except UnicodeDecodeError:
            encoding = 'latin-1'
            content = raw_bytes.decode(encoding)

        content = content.replace('\r\n', '\n').replace('\r', '\n').strip()
        meta, raw_yaml, body, _ = parse_markdown_content(content)
        
        index.documents[str(md_file)] = Document(
            md_file, meta, body.strip(), encoding, fingerprint
        )
        
    return index
