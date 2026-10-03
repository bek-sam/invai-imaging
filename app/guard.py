"""Service auth, heavy-job concurrency limit and render timing (T-9-5, B-19).

A pure ASGI middleware, so both checks run before the request body is read:
- every path except /health needs `X-Imaging-Secret` equal to one of the configured shared
  secrets (constant-time compare), or gets 401;
- heavy paths (compose, personalization render, photo render, zip, scene base and
  scene composite) take a slot from a per-process limiter; with
  none free the request gets 429 + Retry-After at once, before any work, so a retry is safe;
- every response carries `X-Render-Ms`, the time spent handling it.
"""

import hashlib
import hmac
import json
import logging
import time

from app.config import settings

log = logging.getLogger("invai.imaging")

AUTH_HEADER = "x-imaging-secret"
OPEN_PATHS = frozenset({"/health"})
HEAVY_PATHS = frozenset(
    {
        "/compose",
        "/render/personalization",
        "/photo/render",
        "/photo/zip",
        "/photo/scene-base",
        "/photo/scene-composite",
    }
)


def _digest(value: str) -> bytes:
    # Hash first so the constant-time compare doesn't leak the secret's length either.
    return hashlib.sha256(value.encode()).digest()


def secret_ok(provided: str | None) -> bool:
    if not provided:
        return False
    given = _digest(provided)
    ok = False
    for secret in settings.shared_secrets():  # check them all: no early exit on a match
        ok |= hmac.compare_digest(given, _digest(secret))
    return ok


class HeavyJobLimiter:
    """Non-blocking slot counter. All access is on the event loop thread, so no lock needed."""

    def __init__(self) -> None:
        self.in_flight = 0

    def try_acquire(self) -> bool:
        if self.in_flight >= settings.imaging_max_heavy_jobs:
            return False
        self.in_flight += 1
        return True

    def release(self) -> None:
        self.in_flight = max(0, self.in_flight - 1)


limiter = HeavyJobLimiter()


async def _send_json(send, status: int, detail: str, extra: list[tuple[bytes, bytes]] = ()):
    body = json.dumps({"detail": detail}).encode()
    headers = [
        (b"content-type", b"application/json"),
        (b"content-length", str(len(body)).encode()),
        *extra,
    ]
    await send({"type": "http.response.start", "status": status, "headers": headers})
    await send({"type": "http.response.body", "body": body})


class GuardMiddleware:
    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        path = scope["path"]
        if path not in OPEN_PATHS:
            provided = None
            for name, value in scope["headers"]:
                if name == AUTH_HEADER.encode():
                    provided = value.decode("latin-1")
                    break
            if not secret_ok(provided):
                log.warning("imaging auth rejected %s %s", scope["method"], path)
                return await _send_json(send, 401, "missing or invalid imaging credentials")

        heavy = path in HEAVY_PATHS
        if heavy and not limiter.try_acquire():
            retry = str(settings.imaging_retry_after_s).encode()
            log.warning("imaging busy: %s rejected, %d heavy jobs running", path, limiter.in_flight)
            return await _send_json(
                send, 429, "imaging is busy; retry later", [(b"retry-after", retry)]
            )

        started = time.perf_counter()

        async def timed_send(message):
            if message["type"] == "http.response.start":
                ms = round((time.perf_counter() - started) * 1000)
                message = {
                    **message,
                    "headers": [*message.get("headers", []), (b"x-render-ms", str(ms).encode())],
                }
            await send(message)

        try:
            await self.app(scope, receive, timed_send)
        finally:
            if heavy:
                limiter.release()
