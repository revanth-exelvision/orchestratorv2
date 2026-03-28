from __future__ import annotations

import logging
import sys
from typing import Final

_CONFIGURED: bool = False

_ROOT_NAME: Final = "orchestrator"


def get_logger(name: str) -> logging.Logger:
    """Return a logger under the ``orchestrator`` namespace."""
    if not name.startswith(_ROOT_NAME):
        name = f"{_ROOT_NAME}.{name}"
    return logging.getLogger(name)


def configure_logging() -> None:
    """Configure orchestrator logging once (stderr, level from settings)."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    from orchestrator.config import get_settings

    s = get_settings()
    level = getattr(logging, str(s.log_level).upper(), logging.INFO)

    root = logging.getLogger(_ROOT_NAME)
    root.setLevel(level)
    root.handlers.clear()

    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(level)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(levelname)s [%(name)s] %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S%z",
        )
    )
    root.addHandler(handler)
    root.propagate = False

    _CONFIGURED = True
