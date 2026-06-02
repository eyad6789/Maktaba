"""Test fixtures/shims. Stub `fitz` when PyMuPDF isn't installed so modules that
import it at top (ingest.classify/extract) can still be collected for the pure
logic tests. On a machine with PyMuPDF installed this stub is not used."""

from __future__ import annotations

import importlib.util
import sys
import types

if importlib.util.find_spec("fitz") is None and "fitz" not in sys.modules:
    sys.modules["fitz"] = types.ModuleType("fitz")
