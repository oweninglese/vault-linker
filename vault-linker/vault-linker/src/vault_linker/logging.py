from __future__ import annotations

import logging
import sys

def get_logger(name: str = "vault_linker") -> logging.Logger:
    log = logging.getLogger(name)
    if log.handlers:
        return log
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    log.addHandler(h)
    log.setLevel(logging.INFO)
    return log
