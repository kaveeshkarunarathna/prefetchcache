"""Runnable demo: coffee-shop API + PrefetchCache middleware + dashboard.

    uvicorn prefetchcache.demo.app:app --port 8000

Environment:
    MODELS_DIR       directory with one sub-folder per exported model (default models/coffee)
    REDIS_URL        e.g. redis://localhost:6379/0; an in-process store is used if unset
    POLICY           linked (default), fixed or off
    THRESHOLD        confidence threshold for the fixed policy (default 0.5)
    BUDGET_BYTES_PER_SECOND, BUDGET_BURST_BYTES   token bucket (default 20000 / 200000)
    TTL_SECONDS      cache expiry (default 60)
"""
from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, RedirectResponse

from ..middleware.app import install_prefetchcache
from ..middleware.store import MemoryStore, RedisStore
from ..policy.budget import TokenBucket
from ..policy.gate import PrefetchPolicy
from ..policy.linked import BudgetLinkedPolicy
from ..predictors.onnx_predictor import OnnxPredictor
from . import backend

DASHBOARD = Path(__file__).with_name("dashboard.html")


def load_predictors(models_dir: str | Path) -> dict[str, OnnxPredictor]:
    found = {p.parent.name: OnnxPredictor(p.parent) for p in sorted(Path(models_dir).glob("*/model.onnx"))}
    if not found:
        raise RuntimeError(f"no exported models under {models_dir}")
    vocabularies = {tuple(p.meta["vocab"]) for p in found.values()}
    if len(vocabularies) != 1:
        raise RuntimeError("all models must share one endpoint vocabulary")
    return found


def make_store():
    url = os.environ.get("REDIS_URL")
    if not url:
        return MemoryStore()
    import redis.asyncio as redis

    return RedisStore(redis.from_url(url))


def make_policy():
    kind = os.environ.get("POLICY", "linked")
    if kind == "linked":
        return BudgetLinkedPolicy(top_k=1)
    if kind == "fixed":
        return PrefetchPolicy(1, float(os.environ.get("THRESHOLD", "0.5")))
    if kind == "off":
        return PrefetchPolicy(0)
    raise RuntimeError("POLICY must be linked, fixed or off")


def create_app() -> FastAPI:
    app = FastAPI(title="PrefetchCache demo")
    app.include_router(backend.router)
    predictors = load_predictors(os.environ.get("MODELS_DIR", "models/coffee"))
    first = next(iter(predictors.values()))
    bucket = TokenBucket(
        float(os.environ.get("BUDGET_BYTES_PER_SECOND", "20000")),
        float(os.environ.get("BUDGET_BURST_BYTES", "200000")),
        start_time=__import__("time").monotonic(),
    )
    install_prefetchcache(
        app,
        store=make_store(),
        predictors=predictors,
        vocab=first.meta["vocab"],
        model=os.environ.get("MODEL", "lstm" if "lstm" in predictors else next(iter(predictors))),
        policy=make_policy(),
        bucket=bucket,
        ttl_seconds=float(os.environ.get("TTL_SECONDS", "60")),
    )

    @app.get("/", include_in_schema=False)
    async def root():
        return RedirectResponse("/dashboard")

    @app.get("/dashboard", include_in_schema=False)
    async def dashboard():
        return FileResponse(DASHBOARD, media_type="text/html")

    @app.get("/prefetchcache/sizes")
    async def sizes() -> dict[str, int]:
        return backend.page_sizes()

    return app


app = create_app() if os.environ.get("PREFETCHCACHE_NO_AUTOSTART") != "1" else None
