# Imaging service (FastAPI + pyvips). Multi-stage: uv resolves the locked, production-only
# environment in a build stage; the runtime stage has Python and the venv, no uv, no build tools.
# Every base image is pinned by digest (`docker buildx imagetools inspect <image>:<tag>` to bump).

# ---- build: locked production dependencies into /app/.venv --------------------------------
FROM python:3.13-slim@sha256:70729b46c69b4f1e97c4822c1af3df53a1476cf5ddc6c087c0c10bc3a5678c2f AS build
COPY --from=ghcr.io/astral-sh/uv:0.11.17@sha256:03bdc89bb9798628846e60c3a9ad19006c8c3c724ccd2985a33145c039a0577b /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# ---- runtime ------------------------------------------------------------------------------
FROM python:3.13-slim@sha256:70729b46c69b4f1e97c4822c1af3df53a1476cf5ddc6c087c0c10bc3a5678c2f
ENV PYTHONUNBUFFERED=1
WORKDIR /app
RUN groupadd --system --gid 10001 imaging && useradd --system --uid 10001 --gid 10001 --no-create-home --shell /usr/sbin/nologin imaging
COPY --from=build /app/.venv ./.venv
COPY app ./app
# libvips comes from the pyvips[binary] wheel; fonts are bundled in app/fonts.
# The image is the production artifact: it refuses to start without a real
# IMAGING_SHARED_SECRET (32+ chars) and keeps /labels/mock and /sample-art off.
ENV IMAGING_ENV=production IMAGING_DEV_ENDPOINTS=false
USER 10001:10001
EXPOSE 8000
# /health is the one path without the shared secret. slim has no curl, so use Python.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD [".venv/bin/python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"]
# On SIGTERM, stop accepting and give running composes up to 30s to finish (set the
# orchestrator's stop timeout above this, e.g. ECS stopTimeout 45).
CMD [".venv/bin/uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-graceful-shutdown", "30"]
