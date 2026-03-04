
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sqlite3
import yaml

from .config import Config
from .index import (
    ensure_db,
    iter_markdown_files,
    stat_file,
    read_and_hash,
    get_cached_file,
    upsert_file,
    replace_mentions,
    get_backlinks,
)
from .parse import load_note
from .tag_vocab import load_terms_from_tagfile
from .aho import AhoCorasick
from .linker import link_first_per_file
from .meta import infer_title, extract_author, extract_source, mtime_rfc2822_utc, merge_tags
from .hubs import update_hub_page, scrub_all_hubs
from .infer import InferConfig, infer_candidates, aggregate_candidates


def _has_frontmatter(text: str) -> bool:
    return text.startswith("---\n") or text.startswith("---\r\n")


def _render_frontmatter_preserve(fm: dict) -> str:
    preferred = ["title", "author", "source", "created", "tags"]
    out: dict = {}
    for k in preferred:
        if k in fm:
            out[k] = fm[k]
    for k, v in fm.items():
        if k not in out:
            out[k] = v
    y = yaml.safe_dump(out, sort_keys=False, allow_unicode=True, width=88).strip()
    return "---\n" + y + "\n---\n\n"


@dataclass(frozen=True)
class RunStats:
    scanned: int
    processed: int
    wrote_notes: int
    inserted_links: int
    hubs_updated: int
    hubs_scrubbed: int
    terms: int
    candidates_written: int


def run(
    config: Config,
    db_path: Path,
    dry_run: bool = False,
    verbose: bool = False,
    reindex: bool = False,
    discover: bool = False,
    discover_min_count: int = 3,
    discover_out: Path | None = None,
    discover_acronyms: bool = False,
) -> RunStats:
    if config.tagfile is None or not config.tagfile.exists():
        raise SystemExit("tagfile is required and must exist: use --tagfile /path/to/tags.txt")

    terms = load_terms_from_tagfile(config.tagfile)
    matcher = AhoCorasick(terms)

    con = ensure_db(db_path)
    try:
        stats = _run_with_db(
            con,
            config,
            matcher,
            terms,
            dry_run=dry_run,
            verbose=verbose,
            reindex=reindex,
            discover=discover,
            discover_min_count=discover_min_count,
            discover_out=discover_out,
            discover_acronyms=discover_acronyms,
        )
        if not dry_run:
            con.commit()
        else:
            con.rollback()
        return stats
    finally:
        con.close()


def _run_with_db(
    con: sqlite3.Connection,
    config: Config,
    matcher: AhoCorasick,
    terms: list[str],
    dry_run: bool,
    verbose: bool,
    reindex: bool,
    discover: bool,
    discover_min_count: int,
    discover_out: Path | None,
    discover_acronyms: bool,
) -> RunStats:
    vault = config.vault
    files = list(iter_markdown_files(vault, config.ignore_dirs))

    scanned = 0
    processed = 0
    wrote_notes = 0
    inserted_links = 0

    # Collect candidate tags during scan (rel_path, candidates)
    cand_items: list[tuple[str, set[str]]] = []
    infer_cfg = InferConfig(min_len=config.min_term_len, include_acronyms=discover_acronyms)

    for path in files:
        scanned += 1
        rel = str(path.relative_to(vault))

        try:
            mtime_ns, size = stat_file(path)
        except FileNotFoundError:
            continue

        cached = None if reindex else get_cached_file(con, rel)
        if cached and cached[0] == mtime_ns and cached[1] == size:
            # Even if skipping writes, discovery can be expensive; keep it aligned with "processed" files only.
            continue

        processed += 1
        text, sha1 = read_and_hash(path)

        if cached and cached[2] == sha1:
            if not dry_run:
                upsert_file(con, rel, mtime_ns, size, sha1)
            continue

        note = load_note(path)
        body = note.body

        # Approved tag matching (fast): scan once for ALL approved terms
        matches = matcher.find(body)
        found_terms = {t for _, _, t in matches}
        found_terms = {t for t in found_terms if isinstance(t, str) and len(t.strip()) >= config.min_term_len}

        fm = note.frontmatter if isinstance(note.frontmatter, dict) else {}
        fm_before = dict(fm)

        # Article metadata rules
        fm["created"] = mtime_rfc2822_utc(path)
        fm["title"] = infer_title(fm, body, path)

        if not (isinstance(fm.get("author"), str) and fm.get("author").strip()):
            fm["author"] = extract_author(fm, body)
        if not (isinstance(fm.get("source"), str) and fm.get("source").strip()):
            fm["source"] = extract_source(fm, body)

        fm["tags"] = merge_tags(fm.get("tags"), found_terms, min_len=config.min_term_len)
        fm_changed = fm != fm_before

        # Link first occurrence per approved term
        new_body = body
        links_added = 0
        if config.link_bodies and matches:
            new_body, _found2, links_added = link_first_per_file(body, matches)

        new_text = note.text
        body_changed = (new_body != body)

        if fm_changed or not _has_frontmatter(note.text):
            new_text = _render_frontmatter_preserve(fm) + new_body
        elif body_changed:
            prefix_len = len(note.text) - len(note.body)
            new_text = note.text[:prefix_len] + new_body

        if new_text != note.text:
            wrote_notes += 1
            inserted_links += links_added
            if not dry_run:
                path.write_text(new_text, encoding="utf-8")

        if not dry_run:
            upsert_file(con, rel, mtime_ns, size, sha1)
            replace_mentions(con, rel, found_terms)

        # Smart inference (candidates only, NOT hubs) from title + body
        if discover:
            title = fm.get("title") if isinstance(fm.get("title"), str) else path.stem
            cands = infer_candidates(title=title, body=body, cfg=infer_cfg)
            # Do not suggest things already in approved tagfile
            approved = set(t.lower() for t in terms)
            cands = {c for c in cands if c.lower() not in approved}
            if cands:
                cand_items.append((rel, cands))

        if verbose and processed % 200 == 0:
            print(f"[vault-linker] processed={processed} wrote_notes={wrote_notes} links={inserted_links}")

    # Hubs (approved terms only)
    hubs_updated = 0
    for term in sorted(terms, key=lambda s: s.lower()):
        rels = get_backlinks(con, term)
        hub_path = config.hub_path_for(term)
        did = update_hub_page(
            hub_path,
            marker_start=config.hub_marker_start,
            marker_end=config.hub_marker_end,
            term=term,
            rel_paths=rels,
            dry_run=dry_run,
        )
        if did:
            hubs_updated += 1

    hubs_scrubbed = 0
    if not dry_run:
        hub_root = (vault / config.hub_dir) if config.hub_dir is not None else vault
        hubs_scrubbed = scrub_all_hubs(hub_root, config.hub_marker_start, config.hub_marker_end)

    # Write candidate review file
    candidates_written = 0
    if discover and cand_items:
        counts, examples = aggregate_candidates(cand_items)
        out_path = discover_out or (vault / ".vault-linker" / "tag_candidates.txt")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        lines = []
        for term, cnt in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0].lower())):
            if cnt < max(1, discover_min_count):
                continue
            ex = ", ".join(examples.get(term, [])[:3])
            lines.append(f"{term} | {cnt} | {ex}")
            candidates_written += 1

        if not dry_run:
            out_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        if verbose:
            print(f"[vault-linker] candidates written: {candidates_written} -> {out_path}")

    return RunStats(
        scanned=scanned,
        processed=processed,
        wrote_notes=wrote_notes,
        inserted_links=inserted_links,
        hubs_updated=hubs_updated,
        hubs_scrubbed=hubs_scrubbed,
        terms=len(terms),
        candidates_written=candidates_written,
    )
