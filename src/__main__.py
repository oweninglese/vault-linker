import argparse
import sys
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from config import Config
from scanner import scan
from transformer import generate_link_patches
from io_handler import safe_write
from models import ApplyReport
from tags import extract_candidate_tags, write_tags_file, load_approved_tags

console = Console()

def main():
    parser = argparse.ArgumentParser(prog="vault-linker")
    parser.add_argument("vault", help="Path to vault")
    parser.add_argument("--tags", default="tags.csv", help="Path to your tags.csv")
    parser.add_argument("--dry-run", action="store_true")
    
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("tags-scan")
    sub.add_parser("run")

    args = parser.parse_args()
    cfg = Config(dry_run=args.dry_run, tags_file=args.tags)

    console.print(Panel("[bold blue]Vault-Linker 3.0[/bold blue]", subtitle="Safe Linker Engine"))

    with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as prg:
        prg.add_task("Indexing Vault...", total=None)
        idx = scan(args.vault, cfg)
        rep = ApplyReport(files_scanned=len(idx.documents), errors=idx.diagnostics)

        if args.cmd == "tags-scan":
            tags = extract_candidate_tags(idx)
            write_tags_file(cfg.candidates_file, tags)
            console.print(f"[green]Candidates written to {cfg.candidates_file}")
            
        elif args.cmd == "run":
            tags = load_approved_tags(cfg.tags_file)
            if not tags:
                console.print(f"[bold red]Error:[/bold red] {cfg.tags_file} is missing or empty.")
                sys.exit(1)

            task = prg.add_task(f"Linking {len(tags)} tags...", total=len(idx.documents))
            for path, doc in idx.documents.items():
                new_body, count = generate_link_patches(doc.body, tags)
                if count > 0:
                    if not cfg.dry_run:
                        prefix = f"---\n{doc.raw_frontmatter}\n---\n" if doc.raw_frontmatter else ""
                        safe_write(doc.path, prefix + new_body, doc.encoding)
                    rep.files_modified += 1
                    rep.links_created += count
                prg.advance(task)
            console.print(rep.as_rich_table())

if __name__ == "__main__":
    main()
