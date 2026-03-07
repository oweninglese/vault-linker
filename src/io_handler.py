from __future__ import annotations

import os
from pathlib import Path

from .diagnostics import Diagnostic, Severity


def decode_bytes_strict(
    raw: bytes,
    path: Path,
    allowed_encodings: list[str],
) -> tuple[str | None, str | None, Diagnostic | None]:
    last_error: Exception | None = None

    for enc in allowed_encodings:
        try:
            return raw.decode(enc), enc, None
        except UnicodeDecodeError as e:
            last_error = e

    return (
        None,
        None,
        Diagnostic(
            code="ENCODING_INVALID",
            message="Could not decode file using allowed encodings.",
            severity=Severity.ERROR,
            path=str(path),
            context={
                "allowed_encodings": allowed_encodings,
                "last_error": repr(last_error),
            },
        ),
    )


def atomic_write(path: Path, content: str, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        with open(tmp, "wb") as f:
            f.write(content.encode(encoding))
        os.replace(tmp, path)
    except Exception:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise
