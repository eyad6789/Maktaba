"""Tiny test runner so test modules run with plain `python -m tests.test_x`
(no pytest needed). pytest also discovers the `test_*` functions normally."""

from __future__ import annotations

import sys


def run(namespace: dict) -> int:
    fns = {
        k: v
        for k, v in sorted(namespace.items())
        if k.startswith("test_") and callable(v)
    }
    failed = 0
    for name, fn in fns.items():
        try:
            fn()
            print("PASS", name)
        except Exception as exc:  # noqa: BLE001 - report and continue
            failed += 1
            print("FAIL", name, "->", type(exc).__name__ + ":", exc)
    print(f"{len(fns) - failed}/{len(fns)} passed")
    return 1 if failed else 0


def main(namespace: dict) -> None:
    sys.exit(run(namespace))
