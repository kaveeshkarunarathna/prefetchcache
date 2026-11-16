"""Loader for the coffee-shop mobile app clickstreams (Büyüktanır, 2022).

File: ``dataset_session-based_isoweekofday_hour`` (SPMF sequence format, one
sequence per line, items separated by ``-1``, line terminated by ``-2``).

Each item is a 14-digit token::

    UUUUUUU PPP DD HH
    user    page ISO weekday (01-07)  hour (00-23)

Every line belongs to one user. A line is not a single visit: more than half
of the lines contain jumps of several hours, and about a quarter go back in
weekday/hour order. The only time information is the weekday and hour of each
request, so a line is split into visits wherever consecutive requests are not
in the same or the next hour.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

from .sessions import RawSession

TOKEN_LENGTH = 14
HOURS_PER_WEEK = 7 * 24


def parse_token(token: str) -> tuple[str, int, int]:
    """Return (user id, page id, hour of week 0-167)."""
    if len(token) != TOKEN_LENGTH:
        raise ValueError(f"expected a {TOKEN_LENGTH}-digit token, got {token!r}")
    return token[:7], int(token[7:10]), (int(token[10:12]) - 1) * 24 + int(token[12:14])


def page_endpoint(page_id: int) -> str:
    """Map an app page to the API endpoint that serves it.

    One GET endpoint per page until the demo app's routes define the mapping.
    """
    return f"GET /pages/{page_id}"


def split_visits(hours: list[int]) -> list[tuple[int, int]]:
    """Index ranges of visits: a new visit starts unless the request falls in
    the same hour as the previous one or the hour after it."""
    bounds, start = [], 0
    for i in range(1, len(hours)):
        if (hours[i] - hours[i - 1]) % HOURS_PER_WEEK > 1:
            bounds.append((start, i))
            start = i
    bounds.append((start, len(hours)))
    return bounds


def collapse_repeats(pages: list[int]) -> list[int]:
    """Drop immediate repeats of the same page (refreshes and polling)."""
    return [p for i, p in enumerate(pages) if i == 0 or p != pages[i - 1]]


def iter_coffee_sessions(
    path: str | Path,
    split_on_time_gap: bool = True,
    collapse_repeated_pages: bool = True,
    min_length: int = 2,
    max_length: int | None = 200,
) -> Iterator[RawSession]:
    """Yield one RawSession per visit.

    Visits longer than ``max_length`` requests are dropped as automated or
    test traffic. ``times`` is None: there are no per-request timestamps.
    """
    endpoint = {p: page_endpoint(p) for p in range(1000)}
    with open(path, encoding="ascii") as fh:
        for line_no, line in enumerate(fh):
            tokens = line.split()[:-1:2]  # drop the -1 separators and the final -2
            if not tokens:
                continue
            user = tokens[0][:7]
            pages = [int(t[7:10]) for t in tokens]
            if split_on_time_gap:
                hours = [(int(t[10:12]) - 1) * 24 + int(t[12:14]) for t in tokens]
                bounds = split_visits(hours)
            else:
                bounds = [(0, len(pages))]
            for k, (a, b) in enumerate(bounds):
                visit = pages[a:b]
                if collapse_repeated_pages:
                    visit = collapse_repeats(visit)
                if len(visit) < min_length or (max_length is not None and len(visit) > max_length):
                    continue
                yield RawSession(user, f"coffee-{line_no}-{k}", tuple(endpoint[p] for p in visit), None)


def load_coffee(path: str | Path, **options) -> list[RawSession]:
    return list(iter_coffee_sessions(path, **options))
