# invai-imaging

Python service for everything image-heavy:

| Endpoint / job | Does |
| --- | --- |
| `POST /nest` | Packs design boxes onto a 22-inch roll (rectpack, rotation, 0.25in spacing). **Working.** |
| `POST /render/personalization` | SVG template + buyer answers -> 300 DPI transparent PNG |
| `POST /compose` or BullMQ `render` queue | Builds the gang sheet with pyvips, with order number + QR beside each design |
| `POST /qa` | DPI, transparency and line-thickness checks |
| `POST /mockups` | Design onto real blank photos |
| `app/gpu.py` | Modal (Real-ESRGAN, BiRefNet) and fal (Ideogram 3, Recraft) calls |

Called by `invai-backend` workers over HTTP for short jobs; long renders are consumed straight from the BullMQ `render` queue (`app/worker.py`).

```
uv sync
uv run uvicorn app.main:app --reload --port 8000
uv run pytest
```

pyvips is installed with bundled libvips (`pyvips[binary]`), so no system packages are needed.
