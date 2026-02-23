from __future__ import annotations

from dataclasses import dataclass

from vault_linker.config import BuildConfig
from vault_linker.logging import get_logger
from vault_linker.io.fs import ensure_dir, read_text_utf8, relpath_under, write_text_utf8
from vault_linker.core.tags import normalize_tag_list, read_tags_file
from vault_linker.core.yaml import dump_frontmatter
from vault_linker.stages.cache import load_cache, save_cache, should_process, update_cache, clear_cache
from vault_linker.stages.ingest import ingest
from vault_linker.stages.normalize import normalize
from vault_linker.stages.yaml_parse import parse
from vault_linker.stages.infer_tags import infer_tags
from vault_linker.stages.link_body import link_body
from vault_linker.stages.hubs import add_refs, build_hub_index, write_hubs
from vault_linker.stages.metadata import extract_metadata
from vault_linker.stages.years import extract_year_tags

log = get_logger()

@dataclass(frozen=True)
class BuildStats:
    total: int
    processed: int
    skipped_cache: int
    skipped_other: int
    yaml_errors: int

def build(cfg: BuildConfig) -> BuildStats:
    ensure_dir(cfg.output_vault)
    ensure_dir(cfg.cache_dir)

    if cfg.clear_cache and not cfg.dry_run:
        clear_cache(cfg.cache_dir)

    cache = load_cache(cfg.cache_dir)
    registry = read_tags_file(cfg.tags_file)
    if not registry:
        log.info(f"tag registry empty or missing: {cfg.tags_file}")

    hub_index = build_hub_index()
    items = ingest(cfg.input_vault)

    processed = 0
    skipped_cache = 0
    skipped_other = 0
    yaml_errors = 0

    def nullish(v) -> bool:
        return v is None or (isinstance(v, str) and v.strip().lower() in {"null", "none", ""})

    def is_meta_file(rel_str: str) -> bool:
        """
        Files that should NOT participate in tagging/hubbing/linking because they
        contain vocab dumps or generated hubs.
        """
        s = rel_str.strip().lower()
        name = s.split("/")[-1]
        stem = name[:-3] if name.endswith(".md") else name

        if stem in {"tags", "tagfile", "tagsfile", "vocab"}:
            return True
        # hub pages themselves (if they exist in input)
        if name in {"_hubs.md"}:
            return True
        if s.startswith("_hubs/"):
            return True
        return False

    for item in items:
        rel = relpath_under(cfg.input_vault, item.path)
        rel_str = str(rel.as_posix())
        filename_stem = item.path.stem

        try:
            raw = read_text_utf8(item.path)
        except UnicodeDecodeError:
            log.error(f"Non-UTF8 rejected: {rel_str}")
            skipped_other += 1
            continue

        normalized = normalize(raw)

        if not cfg.force and not should_process(rel_str, normalized, cache):
            skipped_cache += 1
            continue

        try:
            fm = parse(normalized)
        except ValueError as e:
            log.error(f"YAML error in {rel_str}: {e}")
            yaml_errors += 1
            skipped_other += 1
            continue

        # ---- Metadata salvage ----
        meta = extract_metadata(filename_stem, fm.body)
        if nullish(fm.data.get("title", None)) and meta.title:
            fm.data["title"] = meta.title
        if nullish(fm.data.get("author", None)) and meta.author:
            fm.data["author"] = meta.author
        if nullish(fm.data.get("source", None)) and meta.source:
            fm.data["source"] = meta.source
        if nullish(fm.data.get("created", None)) and meta.created:
            fm.data["created"] = meta.created

        # Always ensure tags key exists
        fm.data["tags"] = normalize_tag_list(fm.data.get("tags", None))

        # ---- Meta files: write cleaned YAML but do not tag/link/hub ----
        if is_meta_file(rel_str):
            out_text = dump_frontmatter(fm.data) + fm.body.lstrip("\n")
            out_path = (cfg.output_vault / rel).resolve()
            if cfg.dry_run:
                log.info(f"[dry-run] would write meta: {rel_str}")
            else:
                write_text_utf8(out_path, out_text)
                update_cache(rel_str, normalized, cache)
            processed += 1
            continue

        title_for_tagging = str(fm.data.get("title", filename_stem))

        # ---- Curated registry tag matches ----
        res = infer_tags(
            title=title_for_tagging,
            body=fm.body,
            registry=registry,
            yaml_tags=fm.data.get("tags", None),
            vocab_only=cfg.vocab_only,
        )
        tags = normalize_tag_list(res.tags)

        # ---- Add year tags from body ----
        year_tags = extract_year_tags(fm.body)
        tags = normalize_tag_list(tags + year_tags)

        fm.data["tags"] = tags

        # ---- Linking ----
        body_out = fm.body
        if not cfg.no_links and tags:
            body_out = link_body(body_out, tags)

        out_text = dump_frontmatter(fm.data) + body_out.lstrip("\n")
        out_path = (cfg.output_vault / rel).resolve()

        if cfg.dry_run:
            log.info(f"[dry-run] would write: {rel_str}")
        else:
            write_text_utf8(out_path, out_text)
            update_cache(rel_str, normalized, cache)

        # ---- Hub refs with YAML title ----
        add_refs(hub_index, rel_str, tags, title=str(fm.data.get("title", filename_stem)))

        processed += 1

    if not cfg.dry_run:
        write_hubs(cfg.output_vault, hub_index)
        save_cache(cfg.cache_dir, cache)

    return BuildStats(
        total=len(items),
        processed=processed,
        skipped_cache=skipped_cache,
        skipped_other=skipped_other,
        yaml_errors=yaml_errors,
    )
