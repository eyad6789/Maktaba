"""Functional tests for page classification (native / scanned / empty).

Uses a duck-typed fake page so no real PDF (or PyMuPDF) is required.
"""

from __future__ import annotations

import importlib.util
import sys
import types

# Stub fitz when PyMuPDF isn't installed, so ingest.classify imports.
if importlib.util.find_spec("fitz") is None and "fitz" not in sys.modules:
    sys.modules["fitz"] = types.ModuleType("fitz")

from core.models import PageKind  # noqa: E402
from ingest.classify import classify_page  # noqa: E402


class FakePage:
    def __init__(self, text: str = "", images=()):
        self._text = text
        self._images = list(images)

    def get_text(self, *_args, **_kwargs) -> str:
        return self._text

    def get_images(self, full: bool = False):
        return self._images


def test_native_when_enough_text():
    assert classify_page(FakePage(text="x" * 150), min_chars=100) is PageKind.NATIVE


def test_scanned_when_image_and_no_text():
    assert classify_page(FakePage(text="", images=[(1, 2, 3)]), min_chars=100) is PageKind.SCANNED


def test_scanned_when_short_text_but_image():
    assert classify_page(FakePage(text="hi", images=[(1,)]), min_chars=100) is PageKind.SCANNED


def test_empty_when_no_text_no_image():
    assert classify_page(FakePage(text="", images=[]), min_chars=100) is PageKind.EMPTY


def test_min_chars_threshold_boundary():
    assert classify_page(FakePage(text="a" * 100), min_chars=100) is PageKind.NATIVE
    assert classify_page(FakePage(text="a" * 99, images=[(1,)]), min_chars=100) is PageKind.SCANNED


if __name__ == "__main__":
    from tests._runner import main

    main(globals())
