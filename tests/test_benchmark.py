"""Unit tests for the benchmark projection math (pure, no heavy deps)."""

from __future__ import annotations

from scripts.benchmark import BookTiming, _fmt_duration, _percentile, extrapolate


def _native(seconds: float, pages: int) -> BookTiming:
    return BookTiming("n.pdf", seconds, pages, pages, 0, pages, "completed")


def _scanned(seconds: float, pages: int) -> BookTiming:
    return BookTiming("s.pdf", seconds, pages, 0, pages, pages, "completed")


def test_separate_native_and_scanned_rates():
    timings = [_native(10.0, 100), _scanned(50.0, 10)]  # 0.1 s/pg native, 5.0 s/pg scanned
    p = extrapolate(timings, total_books=1000, pages_per_book=400, scanned_frac=0.30, workers=4)
    assert abs(p.native_rate - 0.1) < 1e-9
    assert abs(p.scanned_rate - 5.0) < 1e-9
    # 280k native pages * 0.1 + 120k scanned pages * 5.0 = 628000s single-worker
    assert abs(p.est_seconds_single - 628000.0) < 1.0
    assert abs(p.est_seconds_parallel - 157000.0) < 1.0


def test_overall_rate_fallback_when_no_pure_sample():
    # Only a mixed book: no pure-native or pure-scanned sample -> overall fallback.
    mixed = BookTiming("m.pdf", 20.0, 100, 50, 50, 80, "completed")
    p = extrapolate([mixed], total_books=10, pages_per_book=100, scanned_frac=0.5)
    assert p.native_rate is None and p.scanned_rate is None
    assert abs(p.overall_rate - 0.2) < 1e-9
    assert abs(p.est_seconds_single - (1000 * 0.2)) < 1e-6  # 10*100 pages * 0.2


def test_sec_per_page():
    assert abs(_native(10.0, 100).sec_per_page - 0.1) < 1e-9
    assert _native(10.0, 0).sec_per_page == 0.0


def test_fmt_duration_units():
    assert _fmt_duration(30) == "30.0s"
    assert _fmt_duration(90) == "1.5m"
    assert _fmt_duration(3600) == "1.0h"
    assert _fmt_duration(90000) == "1.0d"


def test_percentile_monotonic():
    vals = [0.1, 0.2, 0.3, 0.4, 0.5]
    assert _percentile(vals, 50) <= _percentile(vals, 95)
    assert _percentile([], 50) == 0.0


if __name__ == "__main__":
    from tests._runner import main

    main(globals())
