"""Unit tests for the golden eval-set generator's pure helpers: stratified
chunk sampling, question-language picking, the quality filter, and the
golden-record builder.

CI-safe: importing ``scripts.gen_eval`` pulls no qdrant/LLM/embedder deps, and
every function exercised here is pure — no network, no model calls.
"""

from __future__ import annotations

from scripts.gen_eval import (
    build_record,
    pick_question_lang,
    quality_ok,
    resolve_chunk_lang,
    sample_chunks,
)

_LONG_TEXT = ("knowledge grows when shared with patient readers everywhere. " * 5).strip()


def _chunk(i: int, *, book: str = "b1", title: str = "Book One",
           text: str | None = None, lang: str = "en", page: int | None = None) -> dict:
    page_start = page if page is not None else i + 1
    return {
        "chunk_id": f"c-{book}-{i}",
        "text": _LONG_TEXT if text is None else text,
        "book_id": book,
        "title": title,
        "author": "Author",
        "page_start": page_start,
        "page_end": page_start + 1,
        "lang": lang,
        "chunk_index": i,
    }


# -- sample_chunks ---------------------------------------------------------------


def test_sample_chunks_stratifies_across_the_book() -> None:
    by_book = {"b1": [_chunk(i) for i in range(30)]}
    out = sample_chunks(by_book, per_book=8, seed=13, min_chars=200)
    indices = [c["chunk_index"] for c in out]
    assert len(indices) == 8
    assert len(set(indices)) == 8                       # one pick per stratum
    assert indices == sorted(indices)                   # stable (title, index) order
    assert indices[0] <= 3 and indices[-1] >= 26        # head and tail both probed
    gaps = [b - a for a, b in zip(indices, indices[1:])]
    assert max(gaps) <= 8                               # evenly spread, no big hole


def test_sample_chunks_is_deterministic_for_a_seed() -> None:
    by_book = {"b1": [_chunk(i) for i in range(30)]}
    first = [c["chunk_id"] for c in sample_chunks(by_book, per_book=8, seed=13, min_chars=200)]
    second = [c["chunk_id"] for c in sample_chunks(by_book, per_book=8, seed=13, min_chars=200)]
    assert first == second


def test_sample_chunks_filters_short_chunks() -> None:
    # Odd indices are too short to be useful eval sources.
    by_book = {"b1": [_chunk(i, text="tiny" if i % 2 else _LONG_TEXT) for i in range(20)]}
    out = sample_chunks(by_book, per_book=5, seed=13, min_chars=200)
    assert len(out) == 5
    assert all(c["chunk_index"] % 2 == 0 for c in out)
    assert all(len(c["text"]) >= 200 for c in out)


def test_sample_chunks_small_book_yields_fewer() -> None:
    by_book = {"b1": [_chunk(i) for i in range(3)]}
    out = sample_chunks(by_book, per_book=8, seed=13, min_chars=200)
    assert [c["chunk_index"] for c in out] == [0, 1, 2]


def test_sample_chunks_orders_multiple_books_by_title_then_index() -> None:
    by_book = {
        "b9": [_chunk(i, book="b9", title="Beta") for i in range(10)],
        "b1": [_chunk(i, book="b1", title="Alpha") for i in range(10)],
    }
    out = sample_chunks(by_book, per_book=2, seed=13, min_chars=200)
    assert [c["title"] for c in out] == ["Alpha", "Alpha", "Beta", "Beta"]
    alpha = [c["chunk_index"] for c in out if c["title"] == "Alpha"]
    beta = [c["chunk_index"] for c in out if c["title"] == "Beta"]
    assert alpha == sorted(alpha) and beta == sorted(beta)


# -- pick_question_lang -------------------------------------------------------------


def test_pick_question_lang_crosses_every_fourth_at_quarter_frac() -> None:
    langs = [pick_question_lang("en", i, 0.25) for i in range(8)]
    assert langs == ["ar", "en", "en", "en", "ar", "en", "en", "en"]


def test_pick_question_lang_never_crosses_at_zero_frac() -> None:
    assert all(pick_question_lang("ar", i, 0.0) == "ar" for i in range(8))
    assert all(pick_question_lang("en", i, 0.0) == "en" for i in range(8))


def test_pick_question_lang_flips_both_directions() -> None:
    assert pick_question_lang("ar", 0, 0.25) == "en"
    assert pick_question_lang("en", 0, 0.25) == "ar"
    assert pick_question_lang("ar", 1, 0.25) == "ar"


# -- resolve_chunk_lang --------------------------------------------------------------


def test_resolve_chunk_lang_passes_through_known_langs() -> None:
    assert resolve_chunk_lang("ar", "whatever") == "ar"
    assert resolve_chunk_lang("en", "whatever") == "en"


def test_resolve_chunk_lang_detects_from_text_when_unknown() -> None:
    assert resolve_chunk_lang("unknown", "A purely English paragraph about books.") == "en"
    assert resolve_chunk_lang("mixed", "فصل كامل عن تاريخ بغداد في العصر العباسي") == "ar"


def test_resolve_chunk_lang_mixed_falls_back_to_majority_script() -> None:
    majority_ar = "المدينة القديمة وتاريخها العريق عبر القرون alpha beta xy"
    majority_en = "the long history of بغداد city in العراق region today overall"
    assert resolve_chunk_lang("mixed", majority_ar) == "ar"
    assert resolve_chunk_lang("mixed", majority_en) == "en"


# -- quality_ok -----------------------------------------------------------------------


def test_quality_ok_accepts_good_english_question() -> None:
    q = "In what year did Ibn Khaldun complete the Muqaddimah?"
    assert quality_ok(q, _chunk(0), "en") is True


def test_quality_ok_accepts_good_arabic_question() -> None:
    q = "في أي عام أكمل ابن خلدون كتابة المقدمة؟"
    assert quality_ok(q, _chunk(0, lang="ar"), "ar") is True


def test_quality_ok_rejects_empty_and_length_extremes() -> None:
    assert quality_ok("", _chunk(0), "en") is False
    assert quality_ok("Why?", _chunk(0), "en") is False               # < 10 chars
    assert quality_ok("What about " + "x" * 200 + "?", _chunk(0), "en") is False


def test_quality_ok_rejects_banned_english_meta_phrases() -> None:
    for phrase in ("the passage", "This Text", "the excerpt",
                   "The Author Writes", "according to the text"):
        q = f"What does {phrase} say about the history of Baghdad markets?"
        assert quality_ok(q, _chunk(0), "en") is False, phrase


def test_quality_ok_rejects_banned_arabic_meta_phrases() -> None:
    for phrase in ("النص", "المقتطف", "الفقرة", "هذا الكتاب يقول"):
        q = f"ماذا يذكر {phrase} عن تاريخ بغداد في العصر العباسي؟"
        assert quality_ok(q, _chunk(0, lang="ar"), "ar") is False, phrase


def test_quality_ok_rejects_wrong_language_question() -> None:
    en_q = "What was the capital of the Abbasid caliphate?"
    ar_q = "ما هي عاصمة الخلافة العباسية في القرن الثاني؟"
    assert quality_ok(en_q, _chunk(0), "ar") is False
    assert quality_ok(ar_q, _chunk(0, lang="ar"), "en") is False


def test_quality_ok_accepts_innocent_words_containing_banned_stems() -> None:
    """النصر/النصف merely contain "النص" — whole-word matching must pass them."""
    q = "في أي معركة حقق الجيش العباسي النصر على الروم في القرن الثالث؟"
    assert quality_ok(q, _chunk(0, lang="ar"), "ar") is True
    q2 = "ما المقصود بالنصف الأول من العصر العباسي عند المؤرخين؟"
    assert quality_ok(q2, _chunk(0, lang="ar"), "ar") is True


def test_quality_ok_accepts_mixed_script_when_majority_matches_target() -> None:
    """Cross-language questions legitimately keep other-script proper nouns."""
    ar_with_latin_noun = "في أي عام نشر Darwin كتابه عن أصل الأنواع حسب المؤلف؟"
    assert quality_ok(ar_with_latin_noun, _chunk(0, lang="en"), "ar") is True
    en_with_arabic_noun = 'What does the term "العصبية" mean in Ibn Khaldun\'s theory?'
    assert quality_ok(en_with_arabic_noun, _chunk(0, lang="ar"), "en") is True


# -- build_record ----------------------------------------------------------------------


def test_build_record_matches_contract_keys_and_values() -> None:
    chunk = _chunk(5, lang="en", page=12)
    rec = build_record(chunk, "What was the first capital of the Abbasids?", "en")
    assert set(rec) == {
        "question", "expect_book_id", "expect_page", "expect_chunk_id",
        "lang", "qtype", "book_title", "source_pages",
    }
    assert rec["question"] == "What was the first capital of the Abbasids?"
    assert rec["expect_book_id"] == "b1"
    assert rec["expect_page"] == chunk["page_start"] == 12
    assert rec["expect_chunk_id"] == chunk["chunk_id"]
    assert rec["lang"] == "en"
    assert rec["book_title"] == "Book One"
    assert rec["source_pages"] == [12, 13]


def test_build_record_qtype_same_vs_cross() -> None:
    en_chunk = _chunk(0, lang="en")
    assert build_record(en_chunk, "A question?", "en")["qtype"] == "same"
    assert build_record(en_chunk, "سؤال عربي؟", "ar")["qtype"] == "cross"
    ar_chunk = _chunk(0, lang="ar", text="نص عربي طويل عن تاريخ المدينة وأهلها")
    assert build_record(ar_chunk, "سؤال عربي؟", "ar")["qtype"] == "same"
    assert build_record(ar_chunk, "A question?", "en")["qtype"] == "cross"


def test_build_record_resolves_mixed_chunk_lang_for_qtype() -> None:
    mixed = _chunk(0, lang="mixed", text="فصل كامل عن تاريخ بغداد في العصر العباسي وما تلاه")
    assert build_record(mixed, "سؤال عربي؟", "ar")["qtype"] == "same"
    assert build_record(mixed, "A question?", "en")["qtype"] == "cross"
