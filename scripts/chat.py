"""Terminal chatbot for the book RAG system.

A thin REPL over the running API's ``/chat`` endpoint — keeps conversation
history, prints grounded answers with their cited sources, and works in Arabic
or English. Uses only the standard library (no extra deps).

Usage
-----
    # with the API running (uvicorn api.main:app)
    python -m scripts.chat
    python -m scripts.chat --url http://localhost:8000 --k 8

Commands inside the chat:  /reset  (clear history)   /exit  (quit)
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request

DEFAULT_URL = "http://localhost:8000"


def _post_chat(url: str, payload: dict, timeout: float = 120.0) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url.rstrip("/") + "/chat", data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _get_status(url: str) -> dict | None:
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/status", timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None


def _format_sources(data: dict) -> str:
    sources = data.get("sources") or []
    if not sources:
        return ""
    # Show the sources cited by [n] markers (fall back to the top few).
    import re

    nums: list[int] = []
    seen = set()
    for m in re.finditer(r"\[(\d+(?:\s*,\s*\d+)*)\]", data.get("answer", "")):
        for tok in m.group(1).split(","):
            n = int(tok.strip())
            if n not in seen:
                seen.add(n)
                nums.append(n)
    if not nums:
        nums = list(range(1, min(3, len(sources)) + 1))

    lines = ["  sources:"]
    for n in nums:
        if 1 <= n <= len(sources):
            s = sources[n - 1]
            pg = (
                f"p.{s['page_start']}"
                if s["page_start"] == s["page_end"]
                else f"pp.{s['page_start']}-{s['page_end']}"
            )
            author = f" — {s['author']}" if s.get("author") else ""
            lines.append(f"    [{n}] {s.get('title') or 'Untitled'}{author} · {pg}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m scripts.chat", description="Chat with your book library.")
    parser.add_argument("--url", default=DEFAULT_URL, help=f"API base URL (default: {DEFAULT_URL})")
    parser.add_argument("--k", type=int, default=None, help="Rerank top-k.")
    parser.add_argument("--model", default=None, help="Override the answer model.")
    parser.add_argument("--book", action="append", dest="book_ids", help="Restrict to a book_id (repeatable).")
    parser.add_argument("--no-condense", action="store_true", help="Do not rewrite follow-ups into a standalone query.")
    args = parser.parse_args(argv)

    status = _get_status(args.url)
    if status is None:
        print(f"⚠  Could not reach the API at {args.url} (is it running?). Continuing anyway.\n")
    else:
        print(f"📚 Connected: {status.get('books', 0)} books · {status.get('chunks', 0)} passages")
    print("Type your question (Arabic or English).  /reset to clear · /exit to quit.\n")

    messages: list[dict] = []
    while True:
        try:
            line = input("you ▸ ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye 👋")
            return 0
        if not line:
            continue
        if line in {"/exit", "/quit"}:
            print("bye 👋")
            return 0
        if line == "/reset":
            messages.clear()
            print("(history cleared)\n")
            continue

        messages.append({"role": "user", "content": line})
        payload: dict = {"messages": messages, "condense": not args.no_condense}
        if args.k:
            payload["top_k"] = args.k
        if args.model:
            payload["model"] = args.model
        if args.book_ids:
            payload["book_ids"] = args.book_ids

        try:
            data = _post_chat(args.url, payload)
        except urllib.error.HTTPError as exc:
            print(f"  ⚠ server error {exc.code}: {exc.read().decode('utf-8', 'replace')[:200]}\n")
            messages.pop()
            continue
        except Exception as exc:  # noqa: BLE001
            print(f"  ⚠ could not reach the server: {exc}\n")
            messages.pop()
            continue

        answer = data.get("answer", "")
        tag = "" if data.get("grounded", True) else "  (not found in the books)"
        print(f"\nlib ▸ {answer}{tag}")
        srcs = _format_sources(data)
        if srcs:
            print(srcs)
        print()
        messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    raise SystemExit(main())
