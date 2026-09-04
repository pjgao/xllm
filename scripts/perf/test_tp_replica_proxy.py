# Copyright 2026 The xLLM Authors. All Rights Reserved.

import asyncio
import importlib.util
from pathlib import Path

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer


_MODULE_PATH = Path(__file__).with_name("tp_replica_proxy.py")
_SPEC = importlib.util.spec_from_file_location("tp_replica_proxy", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def _make_pool() -> object:
    return _MODULE.ReplicaPool(
        ["http://replica-0", "http://replica-1"],
        service_time_seconds=0.32,
        queue_when_all_busy_seconds=0.0,
        release_on_done=True,
        routing_policy="remaining-work",
        batch_penalty_seconds=0.05,
        log_path=None,
    )


def test_overdue_request_remains_more_expensive_than_younger_request() -> None:
    pool = _make_pool()
    pool.started_at[0][0] = 9.6
    pool.started_at[1][1] = 9.9
    scores = pool.scores(now=10.0)
    assert scores[0] > scores[1] > 0.0


def test_health_fails_when_any_backend_is_unavailable() -> None:
    async def run_test() -> None:
        async def ready_handler(request: web.Request) -> web.Response:
            return web.json_response({"status": "ok"})

        backend_app = web.Application()
        backend_app.router.add_get("/health", ready_handler)
        backend_server = TestServer(backend_app)
        await backend_server.start_server()
        proxy_app = await _MODULE.create_app(
            [str(backend_server.make_url("/")).rstrip("/"), "http://127.0.0.1:1"],
            service_time_seconds=0.32,
            queue_when_all_busy_seconds=0.0,
            release_on_done=True,
            routing_policy="remaining-work",
            batch_penalty_seconds=0.05,
            log_path=None,
        )
        client = TestClient(TestServer(proxy_app))
        await client.start_server()
        try:
            response = await client.get("/health")
            assert response.status == 503
            payload = await response.json()
            assert payload["status"] == "unavailable"
            assert [backend["ready"] for backend in payload["backends"]] == [
                True,
                False,
            ]
        finally:
            await client.close()
            await backend_server.close()

    asyncio.run(run_test())
