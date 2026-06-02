"""Functional tests for Arabic/English text normalization."""

from __future__ import annotations

from ingest.normalize import detect_lang, normalize_arabic, normalize_text


def test_alef_variants_fold_to_bare_alef():
    assert normalize_arabic("أإآ") == "ااا"


def test_tatweel_removed_by_default():
    assert normalize_arabic("مـحـمـد") == "محمد"


def test_diacritics_kept_by_default():
    out = normalize_arabic("مُحَمَّد")  # strip_diacritics defaults False
    assert any(0x064B <= ord(ch) <= 0x0652 for ch in out)  # harakat retained


def test_diacritics_stripped_when_requested():
    assert normalize_arabic("مُحَمَّد", strip_diacritics=True) == "محمد"


def test_hamza_carriers_folded():
    assert normalize_arabic("ؤئ") == "وي"


def test_alef_maqsura_folds_to_ya():
    assert normalize_arabic("على") == "علي"


def test_lam_alef_ligature_expanded():
    assert normalize_arabic("ﻻ") == "لا"


def test_teh_marbuta_preserved():
    assert "ة" in normalize_arabic("مدرسة")


def test_normalize_text_collapses_whitespace():
    assert normalize_text("a   b\t c") == "a b c"


def test_normalize_text_strips_zero_width():
    assert normalize_text("a​b") == "ab"


def test_normalize_text_keeps_paragraph_breaks():
    assert normalize_text("para one\n\n\n\npara two") == "para one\n\npara two"


def test_detect_lang_english():
    assert detect_lang("This is a fairly long English sentence here.") == "en"


def test_detect_lang_arabic():
    assert detect_lang("هذا نص عربي طويل بما يكفي للكشف عنه") == "ar"


def test_detect_lang_mixed():
    assert detect_lang("hello world مرحبا بكم في العالم") == "mixed"


def test_detect_lang_unknown():
    assert detect_lang("") == "unknown"
    assert detect_lang("123 456 !!!") == "unknown"


if __name__ == "__main__":
    from tests._runner import main

    main(globals())
