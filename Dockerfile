FROM python:3.13-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PYTHONUNBUFFERED=1
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY app ./app
# libvips comes from the pyvips[binary] wheel; fonts are bundled in app/fonts.
# The image is the production artifact: it refuses to start without a real
# IMAGING_SHARED_SECRET (32+ chars) and keeps /labels/mock and /sample-art off.
ENV IMAGING_ENV=production IMAGING_DEV_ENDPOINTS=false
RUN useradd --system --uid 10001 imaging
USER imaging
EXPOSE 8000
# /health is the one path without the shared secret. slim has no curl, so use Python.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD [".venv/bin/python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"]
# On SIGTERM, stop accepting and give running composes up to 30s to finish (set the
# orchestrator's stop timeout above this, e.g. ECS stopTimeout 45).
CMD [".venv/bin/uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-graceful-shutdown", "30"]
