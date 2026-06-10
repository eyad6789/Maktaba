"""Server-Sent Events framing — stdlib only.

Kept free of FastAPI imports so the framing logic is unit-testable in CI
(which installs only the lightweight dependency set). The streaming endpoint
in :mod:`api.routes_chat` composes its event stream from :func:`format_sse`.
"""

from __future__ import annotations

import json
from typing import Any


def format_sse(event: str, data: dict[str, Any]) -> str:
    """Render one SSE frame: ``event: <name>`` + JSON ``data`` + blank line.

    ``ensure_ascii=False`` keeps Arabic text readable on the wire (SSE is
    UTF-8 by spec). The payload is a single JSON object, so no multi-line
    ``data:`` handling is needed on either side.
    """
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
