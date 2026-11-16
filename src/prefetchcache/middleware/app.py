"""FastAPI integration: an HTTP middleware plus an admin API.

    engine = install_prefetchcache(app, store=..., predictors=..., vocab=..., ...)

GET requests that carry a session id are served from the per-session cache
when possible. The response goes back to the user first; prefetching then
runs as a background task and fetches from the same app through an internal
request that bypasses the cache.
"""
from __future__ import annotations

import asyncio

import httpx
from fastapi import APIRouter, FastAPI, HTTPException, Request, Response
from pydantic import BaseModel

from .engine import PrefetchEngine

SESSION_HEADER = "x-session-id"
INTERNAL_HEADER = "x-prefetchcache-internal"
ADMIN_PREFIX = "/prefetchcache"


class ConfigUpdate(BaseModel):
    model: str | None = None
    policy: str | None = None
    top_k: int | None = None
    threshold: float | None = None
    min_confidence: float | None = None
    budget_bytes_per_second: float | None = None
    budget_burst_bytes: float | None = None
    ttl_seconds: float | None = None


def install_prefetchcache(app: FastAPI, *, store, predictors, vocab, model, policy, bucket=None, ttl_seconds=300.0, clock=None) -> PrefetchEngine:
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://prefetchcache.internal")

    async def fetch_from_backend(endpoint: str) -> bytes:
        response = await client.get(endpoint, headers={INTERNAL_HEADER: "1"})
        response.raise_for_status()
        return response.content

    kwargs = {"clock": clock} if clock else {}
    engine = PrefetchEngine(store, fetch_from_backend, predictors, vocab, model, policy, bucket, ttl_seconds, **kwargs)

    @app.middleware("http")
    async def prefetchcache_middleware(request: Request, call_next):
        session_id = request.headers.get(SESSION_HEADER)
        path = request.url.path
        if (
            request.method != "GET"
            or not session_id
            or INTERNAL_HEADER in request.headers
            or path.startswith(ADMIN_PREFIX)
        ):
            return await call_next(request)
        endpoint = path + (f"?{request.url.query}" if request.url.query else "")
        try:
            cached = await engine.lookup(session_id, endpoint)
        except Exception:  # cache trouble must not break the request (NFR2)
            engine.errors += 1
            return await call_next(request)
        if cached is not None:
            response = Response(cached, media_type="application/json", headers={"x-prefetchcache": "hit"})
        else:
            upstream = await call_next(request)
            body = b"".join([chunk async for chunk in upstream.body_iterator])
            headers = {k: v for k, v in upstream.headers.items() if k.lower() != "content-length"}
            headers["x-prefetchcache"] = "miss"
            response = Response(body, status_code=upstream.status_code, headers=headers, media_type=upstream.media_type)
            if upstream.status_code == 200:
                try:
                    await engine.store_response(session_id, endpoint, body)
                except Exception:
                    engine.errors += 1
        engine.after_response(session_id, endpoint)
        return response

    admin = APIRouter(prefix=ADMIN_PREFIX)

    @admin.get("/stats")
    async def stats() -> dict:
        return engine.stats()

    @admin.get("/config")
    async def get_config() -> dict:
        return engine.config()

    @admin.put("/config")
    async def put_config(update: ConfigUpdate) -> dict:
        try:
            return engine.configure(**update.model_dump(exclude_none=True))
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @admin.get("/log")
    async def prefetch_log(limit: int = 100) -> list[dict]:
        return list(engine.log)[-limit:]

    app.include_router(admin)
    app.state.prefetchcache = engine
    return engine
