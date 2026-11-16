import asyncio

import pytest

fastapi = pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")

from fastapi import FastAPI

from prefetchcache.middleware.app import install_prefetchcache
from prefetchcache.middleware.store import MemoryStore, RedisStore, cache_key
from prefetchcache.policy.budget import TokenBucket
from prefetchcache.policy.gate import PrefetchPolicy

VOCAB = ["<unk>", "GET /pages/1", "GET /pages/2", "GET /pages/3", "POST /cart"]
SIZE = {1: 100, 2: 200, 3: 400}


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class Fixed:
    """After page 1 predict page 2 (0.9); after page 2 predict page 3 (0.6)."""

    def __init__(self, table=None):
        self.table = table or {1: [(2, 0.9)], 2: [(3, 0.6)]}

    def predict(self, history, top_n=5):
        return self.table.get(history[-1], [])


class Broken:
    def predict(self, history, top_n=5):
        raise RuntimeError("model crashed")


def build(policy=PrefetchPolicy(1), bucket=None, predictors=None, store=None, ttl=60.0):
    app = FastAPI()
    calls = {"backend": 0, "fail": False}

    @app.get("/pages/{n}")
    async def page(n: int):
        calls["backend"] += 1
        if calls["fail"]:
            raise RuntimeError("backend down")
        return fastapi.Response(b"x" * SIZE[n], media_type="application/json")

    @app.post("/cart")
    async def cart():
        calls["backend"] += 1
        return {"ok": True}

    clock = Clock()
    engine = install_prefetchcache(
        app, store=store or MemoryStore(clock), predictors=predictors or {"fixed": Fixed()}, vocab=VOCAB,
        model=next(iter(predictors or {"fixed": 0})), policy=policy, bucket=bucket, ttl_seconds=ttl, clock=clock,
    )
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app, raise_app_exceptions=False), base_url="http://test")
    return client, engine, calls, clock


async def get(client, engine, path, session):
    response = await client.get(path, headers={"x-session-id": session})
    await engine.drain()
    return response


async def warm(client, engine):
    """Let the middleware learn response sizes from real requests."""
    for n in SIZE:
        await get(client, engine, f"/pages/{n}", f"warm{n}")
    await engine.drain()
    engine.metrics.__init__()
    engine.log.clear()
    engine._pending.clear()


def run(coro):
    return asyncio.run(coro)


def test_prefetch_is_served_and_logged_as_used():
    async def scenario():
        client, engine, calls, _ = build()
        await warm(client, engine)
        first = await get(client, engine, "/pages/1", "a")
        before = calls["backend"]
        second = await get(client, engine, "/pages/2", "a")
        assert first.headers["x-prefetchcache"] == "miss" and second.headers["x-prefetchcache"] == "hit"
        assert second.content == b"x" * 200 and calls["backend"] == before + 1  # only the prefetch of page 3
        stats = (await client.get("/prefetchcache/stats")).json()
        assert stats["useful_prefetches"] == 1 and stats["useful_bytes"] == 200
        log = (await client.get("/prefetchcache/log")).json()
        assert {"session": "a", "endpoint": "/pages/2", "bytes": 200, "outcome": "used"} in log

    run(scenario())


def test_unused_prefetch_is_logged_as_wasted_after_ttl():
    async def scenario():
        client, engine, _, clock = build(ttl=60.0)
        await warm(client, engine)
        await get(client, engine, "/pages/1", "a")
        clock.now += 61
        stats = engine.stats()
        assert stats["wasted_prefetches"] == 1 and stats["wasted_bytes"] == 200
        assert stats["prefetched_bytes"] == stats["useful_bytes"] + stats["wasted_bytes"] + stats["pending_bytes"]
        assert engine.log[-1]["outcome"] == "wasted" and engine.log[-1]["bytes"] == 200

    run(scenario())


def test_cache_is_isolated_between_sessions():
    # NFR3: a response prefetched or cached for session a is never served to session b.
    async def scenario():
        client, engine, _, _ = build()
        await warm(client, engine)
        await get(client, engine, "/pages/1", "a")  # prefetches page 2 for a
        other = await get(client, engine, "/pages/2", "b")
        assert other.headers["x-prefetchcache"] == "miss"
        assert cache_key("a", "/pages/2") != cache_key("b", "/pages/2")
        assert cache_key("a:b", "c") != cache_key("a", "b:c")

    run(scenario())


def test_requests_without_session_and_posts_bypass_the_cache():
    async def scenario():
        client, engine, _, _ = build()
        await warm(client, engine)
        response = await client.get("/pages/1")
        assert "x-prefetchcache" not in response.headers
        post = await client.post("/cart", headers={"x-session-id": "a"})
        assert post.status_code == 200 and "x-prefetchcache" not in post.headers
        await engine.drain()
        assert engine.metrics.prefetches == 0

    run(scenario())


def test_prefetch_failures_never_reach_the_user():
    # NFR2: a crashing model or a failing prefetch fetch leaves the user's response intact.
    async def scenario():
        client, engine, _, _ = build(predictors={"broken": Broken()})
        response = await get(client, engine, "/pages/1", "a")
        assert response.status_code == 200 and engine.errors == 1

        client, engine, calls, _ = build(bucket=TokenBucket(0.0, 1000.0))

        async def failing_fetch(endpoint):
            raise RuntimeError("backend down")

        await warm(client, engine)
        engine.fetcher = failing_fetch
        response = await get(client, engine, "/pages/1", "a")
        assert response.status_code == 200 and response.content == b"x" * 100
        assert engine.errors == 1 and engine.metrics.prefetches == 0
        assert engine.bucket.tokens == 1000.0  # bytes not transferred are refunded

    run(scenario())


def test_budget_is_never_exceeded():
    # FR3: prefetched bytes stay within burst + rate * elapsed time.
    async def scenario():
        rate, burst = 10.0, 250.0
        table = {1: [(2, 0.9)], 2: [(3, 0.9)], 3: [(1, 0.9)]}
        client, engine, _, clock = build(bucket=TokenBucket(rate, burst), predictors={"fixed": Fixed(table)})
        await warm(client, engine)
        for i in range(200):
            clock.now += 1.0
            await get(client, engine, f"/pages/{1 + i % 3}", f"s{i}")
            assert engine.metrics.prefetched_bytes <= burst + rate * clock.now
        assert engine.metrics.budget_skipped > 0 and engine.metrics.prefetches > 0

    run(scenario())


def test_runtime_configuration():
    # FR4: model, threshold and budget change without a restart.
    async def scenario():
        client, engine, _, _ = build(predictors={"fixed": Fixed(), "other": Fixed({1: [(3, 0.99)]})})
        await warm(client, engine)
        config = (await client.put("/prefetchcache/config", json={"threshold": 0.95})).json()
        assert config["threshold"] == 0.95
        await get(client, engine, "/pages/1", "a")
        assert engine.metrics.prefetches == 0 and engine.metrics.gate_skipped == 1
        await client.put("/prefetchcache/config", json={"model": "other"})
        await get(client, engine, "/pages/1", "b")
        assert engine.log == type(engine.log)() and engine._pending[cache_key("b", "/pages/3")].size == 400
        config = (await client.put("/prefetchcache/config", json={"budget_bytes_per_second": 5, "budget_burst_bytes": 50})).json()
        assert config["budget_bytes_per_second"] == 5 and engine.bucket.capacity == 50
        config = (await client.put("/prefetchcache/config", json={"policy": "linked", "min_confidence": 0.2})).json()
        assert config["policy"] == "linked" and config["min_confidence"] == 0.2
        assert (await client.put("/prefetchcache/config", json={"model": "missing"})).status_code == 422
        assert (await client.put("/prefetchcache/config", json={"threshold": 2})).status_code == 422

    run(scenario())


def test_redis_store_round_trip_and_expiry():
    fakeredis = pytest.importorskip("fakeredis")

    async def scenario():
        store = RedisStore(fakeredis.FakeAsyncRedis())
        await store.set("k", b"value", ttl_seconds=0.05)
        assert await store.get("k") == b"value"
        await asyncio.sleep(0.08)
        assert await store.get("k") is None
        client, engine, _, _ = build(store=RedisStore(fakeredis.FakeAsyncRedis()))
        await warm(client, engine)
        await get(client, engine, "/pages/1", "a")
        assert (await get(client, engine, "/pages/2", "a")).headers["x-prefetchcache"] == "hit"

    run(scenario())


def test_concurrent_requests_do_not_prefetch_the_same_response_twice():
    async def scenario():
        table = {1: [(3, 0.9)], 2: [(3, 0.9)]}
        client, engine, _, _ = build(predictors={"fixed": Fixed(table)})
        await warm(client, engine)
        await asyncio.gather(*(client.get(f"/pages/{n}", headers={"x-session-id": "a"}) for n in (1, 2, 1, 2)))
        await engine.drain()
        stats = engine.stats()
        assert stats["prefetches"] == 1
        assert stats["prefetched_bytes"] == stats["useful_bytes"] + stats["wasted_bytes"] + stats["pending_bytes"]

    run(scenario())
