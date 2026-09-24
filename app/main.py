from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.nesting import Item, nest, utilization

app = FastAPI(title="invai-imaging")


class NestRequest(BaseModel):
    items: list[dict]
    sheet_width_in: float = 22.0
    spacing_in: float = 0.25
    max_length_in: float = 240.0


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.post("/nest")
def nest_items(req: NestRequest) -> dict:
    items = [Item(str(i["id"]), float(i["width_in"]), float(i["height_in"])) for i in req.items]
    try:
        placements, length = nest(items, req.sheet_width_in, req.spacing_in, req.max_length_in)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    return {
        "length_in": length,
        "utilization": utilization(items, length, req.sheet_width_in),
        "placements": [p.__dict__ for p in placements],
    }


@app.post("/render/personalization")
def render_personalization() -> dict:
    raise HTTPException(501, "not implemented")


@app.post("/compose")
def compose_sheet() -> dict:
    raise HTTPException(501, "not implemented")


@app.post("/qa")
def check_print_file() -> dict:
    raise HTTPException(501, "not implemented")


@app.post("/mockups")
def render_mockup() -> dict:
    raise HTTPException(501, "not implemented")
