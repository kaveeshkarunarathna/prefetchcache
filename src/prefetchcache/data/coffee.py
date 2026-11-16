"""Loader for the coffee-shop mobile app clickstreams (Büyüktanır, 2022).

The dataset is distributed in SPMF sequence format: one sequence per line,
items separated by ``-1`` and the line terminated by ``-2``.

NOTE: this loader has not yet been run against the real file, because Kaggle
cannot be reached from the build environment. The SPMF parsing is standard;
where the user id and day/hour sit in the real file must be confirmed and
passed in through ``meta_parser``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterator

from .sessions import RawSession

# Returns (user_id, remaining SPMF text) for one line.
MetaParser = Callable[[int, str], tuple[str, str]]


def parse_spmf_sequence(text: str) -> list[int]:
    """Parse one SPMF sequence into a flat list of page ids.

    Each itemset in this dataset is a single page view, so itemsets are
    flattened in order.
    """
    pages: list[int] = []
    for tok in text.split():
        if tok == "-2":
            break
        if tok == "-1":
            continue
        pages.append(int(tok))
    return pages


def _line_is_user(line_no: int, text: str) -> tuple[str, str]:
    return f"line{line_no}", text


def page_endpoint(page_id: int) -> str:
    """Map an app page to the API endpoint that serves it.

    Placeholder mapping: one GET endpoint per page. The real 46-page mapping
    comes from the demo app's routes (methods step 1).
    """
    return f"GET /pages/{page_id}"


def iter_coffee_sessions(path: str | Path, meta_parser: MetaParser = _line_is_user) -> Iterator[RawSession]:
    with open(path, encoding="utf-8") as fh:
        for line_no, line in enumerate(fh):
            line = line.strip()
            if not line or line.startswith(("#", "%", "@")):
                continue
            user_id, seq = meta_parser(line_no, line)
            pages = parse_spmf_sequence(seq)
            if len(pages) < 2:
                continue
            yield RawSession(user_id, f"coffee-{line_no}", tuple(page_endpoint(p) for p in pages), None)


def load_coffee(path: str | Path, meta_parser: MetaParser = _line_is_user) -> list[RawSession]:
    return list(iter_coffee_sessions(path, meta_parser))
