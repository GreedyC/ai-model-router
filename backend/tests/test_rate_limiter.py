import pytest
from httpx import AsyncClient, ASGITransport
from main import app
from app.auth.rate_limiter import RateLimiter, rate_limiter
from app.storage.database import init_db
from collections import deque
from importlib import import_module


@pytest.fixture
def memory_clock(monkeypatch):
    module = import_module("app.auth.rate_limiter")
    clock = [100.0]
    monkeypatch.setattr(module, "get_redis", lambda: None)
    monkeypatch.setattr(module.time, "time", lambda: clock[0])
    return clock


@pytest.mark.asyncio
async def test_evict_inactive_keys_and_keep_active_limit(memory_clock):
    limiter = RateLimiter()
    await limiter.check_rate_limit("inactive", 1)
    memory_clock[0] = 130.0
    await limiter.check_rate_limit("active", 1)
    memory_clock[0] = 160.0
    assert await limiter.check_rate_limit("active", 1) == (False, 0, 30)
    assert len(limiter._memory_windows) == 1
    assert ("inactive", 60) not in limiter._memory_windows
    assert list(limiter._memory_windows[("active", 60)]) == [130.0]


@pytest.mark.asyncio
async def test_cleanup_respects_each_window_and_empty_keys(memory_clock):
    limiter = RateLimiter()
    await limiter.check_rate_limit("short", 1, 10)
    await limiter.check_rate_limit("long", 1, 120)
    limiter._memory_windows[("empty", 60)] = deque()
    memory_clock[0] = 160.0
    await limiter.check_rate_limit("trigger", 1)
    assert len(limiter._memory_windows) == 2
    assert ("short", 10) not in limiter._memory_windows
    assert ("empty", 60) not in limiter._memory_windows
    assert await limiter.check_rate_limit("long", 1, 120) == (False, 0, 60)
    memory_clock[0] = 220.0
    await limiter.check_rate_limit("trigger", 1)
    assert ("long", 120) not in limiter._memory_windows


@pytest.mark.asyncio
async def test_same_identifier_has_independent_windows(memory_clock):
    limiter = RateLimiter()
    assert (await limiter.check_rate_limit("user", 1, 60))[0]
    assert (await limiter.check_rate_limit("user", 1, 120))[0]
    memory_clock[0] = 160.0
    assert (await limiter.check_rate_limit("user", 1, 60))[0]
    assert not (await limiter.check_rate_limit("user", 1, 120))[0]


@pytest.mark.asyncio
async def test_cleanup_is_periodic_and_clear_resets_it(memory_clock):
    limiter = RateLimiter()
    await limiter.check_rate_limit("short", 1, 10)
    memory_clock[0] = 120.0
    await limiter.check_rate_limit("other", 1)
    assert len(limiter._memory_windows) == 2
    limiter.clear()
    assert limiter._memory_windows == {}


@pytest.mark.asyncio
async def test_concurrent_requests_preserve_limit(memory_clock):
    import asyncio

    limiter = RateLimiter()
    results = await asyncio.gather(*(limiter.check_rate_limit("user", 3) for _ in range(20)))
    assert sum(allowed for allowed, _, _ in results) == 3
    assert sum(len(timestamps) for timestamps in limiter._memory_windows.values()) == 3


@pytest.mark.asyncio
async def test_redis_failure_uses_memory_cleanup(memory_clock, monkeypatch):
    class FailedRedis:
        def pipeline(self):
            raise ConnectionError("unavailable")

    module = import_module("app.auth.rate_limiter")
    monkeypatch.setattr(module, "get_redis", lambda: FailedRedis())
    limiter = RateLimiter()
    await limiter.check_rate_limit("inactive", 1)
    memory_clock[0] = 160.0
    assert (await limiter.check_rate_limit("active", 1))[0]
    assert len(limiter._memory_windows) == 1


@pytest.mark.asyncio
async def test_cleanup_keeps_live_entries_after_clock_rollback(memory_clock):
    limiter = RateLimiter()
    memory_clock[0] = 360.0
    await limiter.check_rate_limit("user", 2, 120)
    memory_clock[0] = 100.0
    await limiter.check_rate_limit("user", 2, 120)
    memory_clock[0] = 420.0
    assert not (await limiter.check_rate_limit("user", 2, 120))[0]


@pytest.fixture(autouse=True)
async def setup_test_state():
    await init_db()
    rate_limiter.clear()


@pytest.mark.asyncio
async def test_in_memory_rate_limiter_unit():
    rl = RateLimiter()
    identifier = "test_user_unit"
    limit = 3

    # First 3 requests should succeed
    for _ in range(limit):
        allowed, remaining, reset_after = await rl.check_rate_limit(identifier, limit_rpm=limit, window_seconds=60)
        assert allowed is True
        assert reset_after > 0

    # 4th request must be blocked
    allowed, remaining, reset_after = await rl.check_rate_limit(identifier, limit_rpm=limit, window_seconds=60)
    assert allowed is False
    assert remaining == 0
    assert reset_after > 0


@pytest.mark.asyncio
async def test_api_rate_limiter_enforcement_429():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        import uuid
        test_ws_id = f"ws-tight-{uuid.uuid4().hex[:6]}"
        # Create a workspace with a strict rate limit (e.g. 2 requests per minute)
        ws_res = await client.post(
            "/api/workspaces",
            json={"id": test_ws_id, "name": "Strict Rate Limit Workspace", "rate_limit_rpm": 2},
        )
        assert ws_res.status_code == 200

        # Create API key for this workspace
        key_res = await client.post(
            "/api/keys",
            json={"name": "Strict Key", "workspace_id": test_ws_id, "rate_limit_rpm": 2},
        )
        assert key_res.status_code == 200
        api_key = key_res.json()["key"]

        # 1st request -> 200
        res1 = await client.post(
            "/api/route",
            json={"prompt": "First request"},
            headers={"X-API-Key": api_key},
        )
        assert res1.status_code == 200

        # 2nd request -> 200
        res2 = await client.post(
            "/api/route",
            json={"prompt": "Second request"},
            headers={"X-API-Key": api_key},
        )
        assert res2.status_code == 200

        # 3rd request -> 429 Too Many Requests
        res3 = await client.post(
            "/api/route",
            json={"prompt": "Third request - should be blocked"},
            headers={"X-API-Key": api_key},
        )
        assert res3.status_code == 429
        assert "Rate limit exceeded" in res3.json()["detail"]
        assert "Retry-After" in res3.headers
        assert res3.headers["X-RateLimit-Limit"] == "2"
        assert res3.headers["X-RateLimit-Remaining"] == "0"
