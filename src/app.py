import argparse
import json
import pathlib
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn

from config import Config
from scanner import scan
from transformer import worker_process_file
from io_handler import safe_write
from tags import load_approved_tags
from frontmatter import (
    update_frontmatter_tags, 
    stringify_frontmatter, 
    enrich_metadata
)

console = Console()
CACHE_FILE = ".linker_cache.json"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("vault")
    parser.add_argument("--tags", default="tags.csv")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    approved_tags = load_approved_tags(args.tags)
    idx = scan(args.vault, Config(dry_run=False, tags_file=args.tags))
    
    cache = {}
    if not args.force and pathlib.Path(CACHE_FILE).exists():
        try:
            with open(CACHE_FILE, 'r') as f:
                cache = json.load(f)
        except: pass

    new_cache = {}
    mod_count = 0
    skip_count = 0

    with Progress(SpinnerColumn(), TextColumn("{task.description}"), BarColumn(), MofNCompleteColumn(), console=console) as prg:
        task = prg.add_task("Turbo Syncing...", total=len(idx.documents))

        for path_str, doc in idx.documents.items():
            path_obj = pathlib.Path(path_str)
            
            # --- 1. THE TURBO CHECK ---
            # If the file size and mtime match our last run, it is physically impossible 
            # for the content to have changed. SKIP EVERYTHING.
            if not args.force and cache.get(path_str) == doc.fingerprint:
                new_cache[path_str] = doc.fingerprint
                skip_count += 1
                prg.advance(task)
                continue

            # --- 2. THE DEEP SCAN (Only runs if file changed or forced) ---
            new_body, _ = worker_process_file(doc.body, approved_tags)
            meta = enrich_metadata(doc.frontmatter.copy(), path_obj, new_body)
            body_tags = [t for t in approved_tags if f"[[{t.lower()}]]" in new_body.lower()]
            meta = update_frontmatter_tags(meta, body_tags)
            
            proposed_text = (stringify_frontmatter(meta) + new_body).strip() + "\n"
            current_text = (stringify_frontmatter(doc.frontmatter) + doc.body).strip() + "\n"

            if proposed_text == current_text:
                # If content is same but fingerprint changed (e.g. opened and saved without edits)
                new_cache[path_str] = doc.fingerprint
                skip_count += 1
            else:
                safe_write(path_str, proposed_text, doc.encoding)
                mod_count += 1
                # Update fingerprint after write
                new_stat = pathlib.Path(path_str).stat()
                new_cache[path_str] = f"{new_stat.st_size}-{new_stat.st_mtime}"
                if args.verbose:
                    console.log(f"[green]UPDATED:[/green] {path_obj.name}")
            
            prg.advance(task)

    with open(CACHE_FILE, 'w') as f:
        json.dump(new_cache, f)

    console.print(f"\n[bold green]Vault Stabilized![/bold green]")
    console.print(f"Modified: {mod_count} | Quick Skipped: {skip_count}")

if __name__ == "__main__":
    main()
