from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import sqlite3
import time

from .config import Config
from .index import (
    ensure_db,
    iter_markdown_files,
    stat_file,
    read_bytes_and_hash,
    get_cached_file,
    upsert_file,
    replace_mentions,
    get_backlinks,
)
from .io_handler import decode_bytes_strict, atomic_write
from .parse import parse_note_text, render_frontmatter
from .tag_vocab import load_terms_from_tagfile
from .aho import AhoCorasick
from .linker import link_matches
from .unlinker import unlink_approved_wikilinks
from .meta import infer_title, extract_author, extract_source, mtime_rfc2822_utc, merge_tags
from .hubs import update_hub_page, is_managed_hub
from .infer import InferConfig, infer_candidates, aggregate_candidates


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
    diagnostics: int = 0
    elapsed_ms: int = 0


@dataclass(frozen=True)
class UnlinkStats:
    scanned: int
    processed: int
    wrote_notes: int
    removed_links: int
    terms: int
    diagnostics: int = 0
    elapsed_ms: int = 0


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
    t0 = time.perf_counter()

    if config.tags_file is not None and isinstance(config.tags_file, str):
        config.tags_file = Path(config.tags_file)

    if config.tags_file is None or not config.tags_file.exists():
        if not config.allow_missing_tagsfile:
            raise SystemExit("tagfile is required and must exist: use --tagfile /path/to/tags.txt")
        terms: list[str] = []
    else:
        terms = load_terms_from_tagfile(config.tags_file)

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
        if dry_run:
            con.rollback()
        else:
            con.commit()

        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        return RunStats(**{**asdict(stats), "elapsed_ms": elapsed_ms})
    finally:
        con.close()


def unlink(
    config: Config,
    db_path: Path,
    dry_run: bool = False,
    verbose: bool = False,
    reindex: bool = False,
) -> UnlinkStats:
    t0 = time.perf_counter()

    if config.tags_file is not None and isinstance(config.tags_file, str):
        config.tags_file = Path(config.tags_file)

    if config.tags_file is None or not config.tags_file.exists():
        raise SystemExit("tagfile is required and must exist: use --tagfile /path/to/tags.txt")

    terms = load_terms_from_tagfile(config.tags_file)
    con = ensure_db(db_path)

    try:
        stats = _unlink_with_db(
            con,
            config,
            terms,
            dry_run=dry_run,
            verbose=verbose,
            reindex=reindex,
        )
        if dry_run:
            con.rollback()
        else:
            con.commit()

        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        return UnlinkStats(**{**asdict(stats), "elapsed_ms": elapsed_ms})
    finally:
        con.close()


def _render_preserved_invalid_frontmatter(raw_frontmatter: str | None, body: str) -> str:
    if raw_frontmatter is None:
        return body
    return f"---\n{raw_frontmatter}---\n{body.lstrip()}"


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
    diagnostics_count = 0

    cand_items: list[tuple[str, set[str]]] = []
    infer_cfg = InferConfig(min_len=config.min_term_len, include_acronyms=discover_acronyms)

    approved_lower = {t.lower() for t in terms}

    for path in files:
        scanned += 1
        rel = str(path.relative_to(vault))

        try:
            mtime_ns, size = stat_file(path)
        except FileNotFoundError:
            continue

        cached = None if reindex else get_cached_file(con, rel)
        if cached and cached[0] == mtime_ns and cached[1] == size:
            continue

        processed += 1

        raw, sha1 = read_bytes_and_hash(path)

        if cached and cached[2] == sha1:
            if not dry_run:
                upsert_file(con, rel, mtime_ns, size, sha1)
            continue

        text, encoding, diag = decode_bytes_strict(raw, path, config.allowed_encodings)
        if diag is not None:
            diagnostics_count += 1
            if verbose:
                print(f"[vault-linker] skip unreadable: {rel} ({diag.code})")
            continue
        assert text is not None
        assert encoding is not None

        if is_managed_hub(text, config.hub_marker_start, config.hub_marker_end):
            if not dry_run:
                upsert_file(con, rel, mtime_ns, size, sha1)
            if verbose and processed % 200 == 0:
                print(
                    f"[vault-linker] processed={processed} "
                    f"wrote_notes={wrote_notes} links={inserted_links}"
                )
            continue

        note = parse_note_text(path, text, encoding)
        diagnostics_count += len(note.diagnostics)

        matches = matcher.find(note.body) if terms else []
        found_terms = {
            term
            for _, _, term in matches
            if isinstance(term, str) and len(term.strip()) >= config.min_term_len
        }

        new_body = note.body
        links_added = 0
        if config.link_bodies and matches:
            new_body, _, links_added = link_matches(
                note.body,
                matches,
                linkify_mode=config.linkify_mode,
            )

        new_text = note.text

        if note.frontmatter_valid or not note.has_frontmatter or config.repair_frontmatter:
            fm = dict(note.frontmatter)

            # Only set created once, if missing.
            if not (isinstance(fm.get("created"), str) and fm.get("created").strip()):
                fm["created"] = mtime_rfc2822_utc(path)

            fm["title"] = infer_title(fm, note.body, path)

            if not (isinstance(fm.get("author"), str) and fm.get("author").strip()):
                fm["author"] = extract_author(fm, note.body)

            if not (isinstance(fm.get("source"), str) and fm.get("source").strip()):
                fm["source"] = extract_source(fm, note.body)

            fm["tags"] = merge_tags(fm.get("tags"), found_terms, min_len=config.min_term_len)

            new_text = render_frontmatter(fm) + new_body.lstrip("\n")
        else:
            new_text = _render_preserved_invalid_frontmatter(note.raw_frontmatter, new_body)

        if new_text != note.text:
            wrote_notes += 1
            inserted_links += links_added
            if not dry_run:
                atomic_write(path, new_text, encoding="utf-8")
                _, new_sha1 = read_bytes_and_hash(path)
                new_mtime_ns, new_size = stat_file(path)
                upsert_file(con, rel, new_mtime_ns, new_size, new_sha1)
                replace_mentions(con, rel, found_terms)
        else:
            if not dry_run:
                upsert_file(con, rel, mtime_ns, size, sha1)
                replace_mentions(con, rel, found_terms)

        if discover:
            title = path.stem
            if note.frontmatter_valid:
                maybe_title = note.frontmatter.get("title")
                if isinstance(maybe_title, str) and maybe_title.strip():
                    title = maybe_title.strip()

            cands = infer_candidates(title=title, body=note.body, cfg=infer_cfg)
            cands = {c for c in cands if c.lower() not in approved_lower}
            if cands:
                cand_items.append((rel, cands))

        if verbose and processed % 200 == 0:
            print(
                f"[vault-linker] processed={processed} "
                f"wrote_notes={wrote_notes} links={inserted_links}"
            )

    hubs_updated = 0
    for term in sorted(terms, key=lambda s: s.lower()):
        rels = get_backlinks(con, term)
        if not rels:
            continue
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

    candidates_written = 0
    if discover and cand_items:
        counts, examples = aggregate_candidates(cand_items)
        out_path = discover_out or (vault / ".vault-linker" / "tag_candidates.txt")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        lines: list[str] = []
        for term, cnt in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0].lower())):
            if cnt < max(1, discover_min_count):
                continue
            ex = ", ".join(examples.get(term, [])[:3])
            lines.append(f"{term} | {cnt} | {ex}")
            candidates_written += 1

        if not dry_run:
            atomic_write(
                out_path,
                "\n".join(lines) + ("\n" if lines else ""),
                encoding="utf-8",
            )

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
        diagnostics=diagnostics_count,
        elapsed_ms=0,
    )


def _unlink_with_db(
    con: sqlite3.Connection,
    config: Config,
    terms: list[str],
    dry_run: bool,
    verbose: bool,
    reindex: bool,
) -> UnlinkStats:
    vault = config.vault
    files = list(iter_markdown_files(vault, config.ignore_dirs))

    scanned = 0
    processed = 0
    wrote_notes = 0
    removed_links = 0
    diagnostics_count = 0

    for path in files:
        scanned += 1
        rel = str(path.relative_to(vault))

        try:
            mtime_ns, size = stat_file(path)
        except FileNotFoundError:
            continue

        cached = None if reindex else get_cached_file(con, rel)
        if cached and cached[0] == mtime_ns and cached[1] == size:
            continue

        processed += 1

        raw, sha1 = read_bytes_and_hash(path)

        text, encoding, diag = decode_bytes_strict(raw, path, config.allowed_encodings)
        if diag is not None:
            diagnostics_count += 1
            if verbose:
                print(f"[vault-linker] skip unreadable: {rel} ({diag.code})")
            continue
        assert text is not None
        assert encoding is not None

        if is_managed_hub(text, config.hub_marker_start, config.hub_marker_end):
            if not dry_run:
                upsert_file(con, rel, mtime_ns, size, sha1)
            if verbose and processed % 200 == 0:
                print(
                    f"[vault-linker:unlink] processed={processed} "
                    f"wrote_notes={wrote_notes} removed_links={removed_links}"
                )
            continue

        new_text, removed = unlink_approved_wikilinks(text, terms)

        if new_text != text:
            wrote_notes += 1
            removed_links += removed
            if not dry_run:
                atomic_write(path, new_text, encoding="utf-8")
                _, new_sha1 = read_bytes_and_hash(path)
                new_mtime_ns, new_size = stat_file(path)
                upsert_file(con, rel, new_mtime_ns, new_size, new_sha1)
                replace_mentions(con, rel, set())
        else:
            if not dry_run:
                upsert_file(con, rel, mtime_ns, size, sha1)

        if verbose and processed % 200 == 0:
            print(
                f"[vault-linker:unlink] processed={processed} "
                f"wrote_notes={wrote_notes} removed_links={removed_links}"
            )

    return UnlinkStats(
        scanned=scanned,
        processed=processed,
        wrote_notes=wrote_notes,
        removed_links=removed_links,
        terms=len(terms),
        diagnostics=diagnostics_count,
        elapsed_ms=0,
    )
