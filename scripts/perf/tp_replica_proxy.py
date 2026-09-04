#!/usr/bin/env python3
"""Least-inflight streaming reverse proxy for independent xLLM replicas."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from contextlib import suppress

from aiohttp import ClientSession, ClientTimeout, TCPConnector, web


HOP_BY_HOP_HEADERS = {
    "connection",
    "content-length",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}


class ReplicaPool:
    def __init__(
        self,
        backends: list[str],
        service_time_seconds: float,
        queue_when_all_busy_seconds: float,
        release_on_done: bool,
        routing_policy: str,
        batch_penalty_seconds: float,
        log_path: str | None,
    ) -> None:
        self.backends = [backend.rstrip("/") for backend in backends]
        self.inflight = [0] * len(backends)
        self.started_at: list[dict[int, float]] = [dict() for _ in backends]
        self.service_time_seconds = service_time_seconds
        self.queue_when_all_busy_seconds = queue_when_all_busy_seconds
        self.release_on_done = release_on_done
        self.routing_policy = routing_policy
        self.batch_penalty_seconds = batch_penalty_seconds
        self.request_sequence = 0
        self.next_tie = 0
        self.condition = asyncio.Condition()
        self.log_file = open(log_path, "a", buffering=1) if log_path else None

    def scores(self, now: float) -> list[float]:
        remaining = [
            [
                (
                    self.service_time_seconds - (now - started)
                    if now - started < self.service_time_seconds
                    else (
                        self.service_time_seconds
                        if now - started > 4.0 * self.service_time_seconds
                        else 0.0
                    )
                )
                for started in active.values()
            ]
            for active in self.started_at
        ]
        if self.routing_policy == "batch-aware":
            # xLLM executes active requests as a dynamic batch. Treating their
            # remaining times as serial work increasingly over-penalizes a
            # replica as its batch grows. The youngest request approximates
            # the batch tail; the small per-member penalty accounts for the
            # extra decode cost of widening that batch.
            return [
                (max(times) + self.batch_penalty_seconds * len(times))
                if times
                else 0.0
                for times in remaining
            ]
        if self.routing_policy == "least-inflight":
            return [float(len(times)) for times in remaining]
        return [sum(times) for times in remaining]

    async def acquire(self) -> tuple[int, int, int, list[float], float, float]:
        queued_at = time.monotonic()
        deadline = queued_at + self.queue_when_all_busy_seconds
        async with self.condition:
            while True:
                now = time.monotonic()
                scores = self.scores(now)
                if min(scores) <= 0.0 or now >= deadline:
                    break
                wait_seconds = min(deadline - now, min(scores))
                try:
                    await asyncio.wait_for(
                        self.condition.wait(), timeout=wait_seconds
                    )
                except TimeoutError:
                    pass
            now = time.monotonic()
            scores = self.scores(now)
            selection_scores = scores
            if self.routing_policy == "batch-affinity" and all(
                active for active in self.started_at
            ):
                # When both replicas are occupied, co-locate a burst with the
                # most recently started batch. This leaves the older batch on
                # the other replica undisturbed so that it can drain sooner.
                selection_scores = [
                    min(now - started for started in active.values())
                    for active in self.started_at
                ]
            minimum = min(selection_scores)
            candidates = [
                index
                for index, score in enumerate(selection_scores)
                if abs(score - minimum) < 1e-9
            ]
            index = next(
                (
                    candidate
                    for candidate in candidates
                    if candidate >= self.next_tie
                ),
                candidates[0],
            )
            request_sequence = self.request_sequence
            self.request_sequence += 1
            inflight_at_acquire = self.inflight[index]
            self.inflight[index] += 1
            self.started_at[index][request_sequence] = now
            self.next_tie = (index + 1) % len(self.backends)
            return (
                index,
                request_sequence,
                inflight_at_acquire,
                scores,
                now,
                now - queued_at,
            )

    async def release(
        self,
        index: int,
        request_sequence: int,
        inflight_at_acquire: int,
        scores: list[float],
        started: float,
        status: int,
        queue_seconds: float,
        stream_completed: bool,
        error: str | None = None,
    ) -> None:
        async with self.condition:
            self.inflight[index] -= 1
            self.started_at[index].pop(request_sequence, None)
            self.condition.notify_all()
            if self.log_file is not None:
                self.log_file.write(
                    json.dumps(
                        {
                            "request": request_sequence,
                            "replica": index,
                            "inflight_at_acquire": inflight_at_acquire,
                            "scores_ms": [round(score * 1000.0, 3) for score in scores],
                            "backend_ms": round(
                                (time.monotonic() - started) * 1000.0, 3
                            ),
                            "queue_ms": round(queue_seconds * 1000.0, 3),
                            "status": status,
                            "stream_completed": stream_completed,
                            "error": error,
                        },
                        separators=(",", ":"),
                    )
                    + "\n"
                )

    def close(self) -> None:
        if self.log_file is not None:
            self.log_file.close()


def forwarded_headers(headers: web.BaseRequest.headers.__class__) -> dict[str, str]:
    return {
        name: value
        for name, value in headers.items()
        if name.lower() not in HOP_BY_HOP_HEADERS and name.lower() != "host"
    }


async def proxy(request: web.Request) -> web.StreamResponse:
    pool: ReplicaPool = request.app["pool"]
    session: ClientSession = request.app["session"]
    (
        index,
        request_sequence,
        inflight_at_acquire,
        scores,
        started,
        queue_seconds,
    ) = await pool.acquire()
    backend = pool.backends[index]
    target = f"{backend}{request.rel_url}"
    status = 502
    released = False
    stream_completed = False
    error: str | None = None
    try:
        body = await request.read()
        async with session.request(
            request.method,
            target,
            headers=forwarded_headers(request.headers),
            data=body,
        ) as upstream:
            status = upstream.status
            response = web.StreamResponse(
                status=upstream.status,
                reason=upstream.reason,
                headers={
                    name: value
                    for name, value in upstream.headers.items()
                    if name.lower() not in HOP_BY_HOP_HEADERS
                },
            )
            response.headers["x-xllm-replica"] = str(index)
            await response.prepare(request)
            stream_tail = b""
            async for chunk in upstream.content.iter_any():
                await response.write(chunk)
                if (
                    pool.release_on_done
                    and not released
                    and b"data: [DONE]" in stream_tail + chunk
                ):
                    await pool.release(
                        index,
                        request_sequence,
                        inflight_at_acquire,
                        scores,
                        started,
                        status,
                        queue_seconds,
                        stream_completed=True,
                    )
                    released = True
                stream_tail = (stream_tail + chunk)[-32:]
            with suppress(ConnectionResetError):
                await response.write_eof()
            stream_completed = True
            return response
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        error = str(exc)
        raise web.HTTPBadGateway(text=f"xLLM replica {index} failed: {exc}") from exc
    finally:
        if not released:
            await pool.release(
                index,
                request_sequence,
                inflight_at_acquire,
                scores,
                started,
                status,
                queue_seconds,
                stream_completed,
                error,
            )


async def health(request: web.Request) -> web.Response:
    pool: ReplicaPool = request.app["pool"]
    session: ClientSession = request.app["session"]

    async def probe(backend: str) -> dict[str, str | int | bool]:
        try:
            async with session.get(
                f"{backend}/health", timeout=ClientTimeout(total=2.0)
            ) as response:
                return {
                    "backend": backend,
                    "ready": 200 <= response.status < 300,
                    "status": response.status,
                }
        except Exception as exc:
            return {"backend": backend, "ready": False, "error": str(exc)}

    backend_status = await asyncio.gather(
        *(probe(backend) for backend in pool.backends)
    )
    ready = all(bool(status["ready"]) for status in backend_status)
    return web.json_response(
        {
            "status": "ok" if ready else "unavailable",
            "backends": backend_status,
            "inflight": pool.inflight,
        },
        status=200 if ready else 503,
    )


async def create_app(
    backends: list[str],
    service_time_seconds: float,
    queue_when_all_busy_seconds: float,
    release_on_done: bool,
    routing_policy: str,
    batch_penalty_seconds: float,
    log_path: str | None,
) -> web.Application:
    app = web.Application(client_max_size=64 * 1024 * 1024)
    app["pool"] = ReplicaPool(
        backends,
        service_time_seconds,
        queue_when_all_busy_seconds,
        release_on_done,
        routing_policy,
        batch_penalty_seconds,
        log_path,
    )
    app["session"] = ClientSession(
        timeout=ClientTimeout(total=None, connect=10),
        connector=TCPConnector(limit=0, ttl_dns_cache=300),
    )
    app.router.add_get("/health", health)
    app.router.add_route("*", "/{path:.*}", proxy)

    async def close_session(application: web.Application) -> None:
        await application["session"].close()
        application["pool"].close()

    app.on_cleanup.append(close_session)
    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--listen-host", default="127.0.0.1")
    parser.add_argument("--listen-port", type=int, required=True)
    parser.add_argument("--backend", action="append", required=True)
    parser.add_argument(
        "--service-time-ms",
        type=float,
        default=350.0,
        help="Expected isolated request time used to estimate remaining work.",
    )
    parser.add_argument(
        "--release-on-done",
        action="store_true",
        help="Mark a streaming request idle when its SSE [DONE] is forwarded.",
    )
    parser.add_argument(
        "--routing-policy",
        choices=(
            "remaining-work",
            "least-inflight",
            "batch-aware",
            "batch-affinity",
        ),
        default="remaining-work",
        help="Replica selection policy when more than one backend is busy.",
    )
    parser.add_argument(
        "--batch-penalty-ms",
        type=float,
        default=50.0,
        help="Per-active-request cost used by batch-aware routing.",
    )
    parser.add_argument("--log-file", help="Append one JSON record per request.")
    parser.add_argument(
        "--queue-when-all-busy-ms",
        type=float,
        default=0.0,
        help="Wait this long for a replica to become idle before stacking work.",
    )
    args = parser.parse_args()
    web.run_app(
        create_app(
            args.backend,
            args.service_time_ms / 1000.0,
            args.queue_when_all_busy_ms / 1000.0,
            args.release_on_done,
            args.routing_policy,
            args.batch_penalty_ms / 1000.0,
            args.log_file,
        ),
        host=args.listen_host,
        port=args.listen_port,
        access_log=None,
    )


if __name__ == "__main__":
    main()
