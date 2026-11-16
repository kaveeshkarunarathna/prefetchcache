import asyncio
import os
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")

MODELS = Path(__file__).resolve().parents[1] / "models" / "coffee"


@pytest.mark.skipif(not (MODELS / "lstm" / "model.onnx").exists(), reason="exported models not present")
def test_demo_app_serves_pages_prefetches_and_reports(monkeypatch):
    monkeypatch.setenv("MODELS_DIR", str(MODELS))
    monkeypatch.setenv("PREFETCHCACHE_NO_AUTOSTART", "1")
    monkeypatch.delenv("REDIS_URL", raising=False)
    from prefetchcache.demo import backend
    from prefetchcache.demo.app import create_app

    monkeypatch.setattr(backend, "BASE_DELAY_MS", 0.0)
    monkeypatch.setattr(backend, "DELAY_MS_PER_KB", 0.0)

    async def scenario():
        app = create_app()
        engine = app.state.prefetchcache
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            sizes = (await client.get("/prefetchcache/sizes")).json()
            assert len(sizes) == 46 and all(v > 0 for v in sizes.values())
            assert (await client.get("/pages/999", headers={"x-session-id": "s"})).status_code == 404
            assert "PrefetchCache" in (await client.get("/dashboard")).text
            config = (await client.get("/prefetchcache/config")).json()
            assert config["models"] == ["lstm", "transformer"]
            for page in range(46):  # let the middleware learn every response size
                await client.get(f"/pages/{page}", headers={"x-session-id": f"warm{page}"})
            await engine.drain()
            await client.post("/prefetchcache/reset")
            for i in range(30):
                for page in (1, 2, 3, 2, 1):
                    r = await client.get(f"/pages/{page}", headers={"x-session-id": f"user{i}"})
                    assert r.status_code == 200 and len(r.content) == sizes[f"GET /pages/{page}"]
                await engine.drain()
            stats = (await client.get("/prefetchcache/stats")).json()
            assert stats["prefetches"] > 0 and stats["prefetch_errors"] == 0
            assert stats["prefetched_bytes"] == stats["useful_bytes"] + stats["wasted_bytes"] + stats["pending_bytes"]
            assert stats["decision_p95_ms"] < 5.0

    asyncio.run(scenario())
