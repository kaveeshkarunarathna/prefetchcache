"""Demo coffee-shop API: one GET endpoint per app page.

The dataset identifies pages only by number, so each page returns generated
catalogue-style JSON whose size is fixed per page and varies between pages
(roughly 0.1 kB to 18 kB). A simulated backend delay stands in for database
work, so a cache hit is visibly faster than a miss.
"""
from __future__ import annotations

import asyncio
import json
import os

import numpy as np
from fastapi import APIRouter, HTTPException, Response

N_PAGES = 46
BASE_DELAY_MS = float(os.environ.get("BACKEND_BASE_DELAY_MS", "20"))
DELAY_MS_PER_KB = float(os.environ.get("BACKEND_DELAY_MS_PER_KB", "1.5"))

_WORDS = "espresso latte mocha filter cold brew oat milk syrup croissant bagel loyalty reward order pickup store offer".split()


def _build_payload(page: int) -> bytes:
    rng = np.random.default_rng(1000 + page)
    n_items = int(np.clip(rng.lognormal(2.6, 1.1), 1, 400))
    items = [
        {
            "id": page * 1000 + i,
            "name": " ".join(rng.choice(_WORDS, size=3)),
            "price": round(float(rng.uniform(1.5, 9.5)), 2),
            "description": " ".join(rng.choice(_WORDS, size=int(rng.integers(6, 18)))),
        }
        for i in range(n_items)
    ]
    return json.dumps({"page": page, "items": items}).encode()


PAYLOADS = {page: _build_payload(page) for page in range(N_PAGES)}


def page_sizes() -> dict[str, int]:
    """Response size in bytes per endpoint, as served by this app."""
    return {f"GET /pages/{page}": len(body) for page, body in PAYLOADS.items()}


router = APIRouter()


@router.get("/pages/{page}")
async def get_page(page: int) -> Response:
    body = PAYLOADS.get(page)
    if body is None:
        raise HTTPException(status_code=404, detail="unknown page")
    await asyncio.sleep((BASE_DELAY_MS + DELAY_MS_PER_KB * len(body) / 1000) / 1000)
    return Response(body, media_type="application/json")
