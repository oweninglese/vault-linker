from __future__ import annotations

import re


_WIKILINK_RE = re.compile(r"\[\[([^\[\]|#]+?)(?:#[^\]|]+)?(?:\|[^\]]+)?\]\]")


def unlink_approved_wikilinks(text: str, approved_terms: list[str]) -> tuple[str, int]:
    """
    Remove wikilinks only when their target matches an approved term.

    Examples:
      [[Canada]]            -> Canada
      [[Canada|CA]]         -> CA
      [[Canada#History]]    -> Canada
      [[Canada#History|CA]] -> CA

    Non-approved wikilinks are preserved unchanged.
    Matching is case-insensitive against the target portion only.
    """
    approved = {t.strip().lower(): t for t in approved_terms if isinstance(t, str) and t.strip()}
    if not approved:
        return text, 0

    removed = 0

    def repl(match: re.Match[str]) -> str:
        nonlocal removed

        raw = match.group(0)
        inner = raw[2:-2]

        # Split alias if present
        if "|" in inner:
            target_part, alias = inner.split("|", 1)
            alias = alias.strip()
        else:
            target_part, alias = inner, None

        # Strip heading anchor for matching
        if "#" in target_part:
            target_base = target_part.split("#", 1)[0].strip()
        else:
            target_base = target_part.strip()

        if target_base.lower() not in approved:
            return raw

        removed += 1

        # Preserve alias display text if present; otherwise use target base
        if alias is not None and alias != "":
            return alias
        return target_base

    new_text = _WIKILINK_RE.sub(repl, text)
    return new_text, removed
