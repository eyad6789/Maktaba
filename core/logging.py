"""Shared logger factory. Use `get_logger(__name__)` in every module."""

from __future__ import annotations

import logging
import sys

from config import settings

_CONFIGURED = False


def get_logger(name: str) -> logging.Logger:
    global _CONFIGURED
    if not _CONFIGURED:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)-7s %(name)s | %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        root = logging.getLogger()
        root.addHandler(handler)
        root.setLevel(settings.log_level.upper())
        _CONFIGURED = True
    return logging.getLogger(name)
