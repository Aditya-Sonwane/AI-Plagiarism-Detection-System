"""Centralized logging configuration."""
from __future__ import annotations

import logging
import sys

from backend.config import settings

_CONFIGURED = False


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger with consistent formatting."""
    global _CONFIGURED
    if not _CONFIGURED:
        logging.basicConfig(
            level=getattr(logging, settings.log_level.upper(), logging.INFO),
            format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            stream=sys.stdout,
        )
        _CONFIGURED = True
    return logging.getLogger(name)
