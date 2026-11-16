"""Loader for the Retailrocket e-commerce events (events.csv).

Columns: timestamp (ms), visitorid, event (view/addtocart/transaction),
itemid, transactionid. Events are grouped into sessions by an inactivity gap.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .sessions import RawSession

SESSION_GAP_SECONDS = 30 * 60


def event_endpoint(event: str, itemid: int) -> str:
    if event == "view":
        return f"GET /items/{itemid}"
    if event == "addtocart":
        return "POST /cart"
    if event == "transaction":
        return "POST /orders"
    raise ValueError(f"unknown event type: {event}")


def sessions_from_events(df: pd.DataFrame, gap_seconds: float = SESSION_GAP_SECONDS, min_length: int = 2) -> list[RawSession]:
    df = df.sort_values(["visitorid", "timestamp"], kind="stable")
    out: list[RawSession] = []
    for visitor, grp in df.groupby("visitorid", sort=True):
        times = (grp["timestamp"].to_numpy() / 1000.0).tolist()
        eps = [event_endpoint(e, i) for e, i in zip(grp["event"], grp["itemid"])]
        start, n = 0, 0
        for i in range(1, len(times) + 1):
            if i == len(times) or times[i] - times[i - 1] > gap_seconds:
                if i - start >= min_length:
                    out.append(RawSession(str(visitor), f"rr-{visitor}-{n}", tuple(eps[start:i]), tuple(times[start:i])))
                    n += 1
                start = i
    return out


def load_retailrocket(path: str | Path, gap_seconds: float = SESSION_GAP_SECONDS, min_length: int = 2) -> list[RawSession]:
    df = pd.read_csv(path, usecols=["timestamp", "visitorid", "event", "itemid"])
    return sessions_from_events(df, gap_seconds, min_length)
