"""Mock 4x6-inch shipping label PDF for the sandbox carrier, clearly watermarked as a sample."""

import io
from dataclasses import dataclass

from reportlab.graphics.barcode import code128
from reportlab.lib.colors import Color, black, white
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas

W, H = 4 * inch, 6 * inch
PAD = 0.16 * inch


@dataclass
class Address:
    name: str
    city: str
    state: str
    zip: str


def _fit(c: canvas.Canvas, text: str, font: str, size: float, max_w: float) -> float:
    while size > 5 and c.stringWidth(text, font, size) > max_w:
        size -= 0.5
    return size


def render_label(
    shipment_id: str,
    carrier: str,
    service: str,
    tracking_code: str,
    sender: Address,
    recipient: Address,
    weight_oz: float,
) -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=(W, H))
    c.setTitle(f"Sample label {tracking_code}")
    c.setAuthor("InvAI")
    inner = W - 2 * PAD

    c.setLineWidth(1.5)
    c.rect(PAD / 2, PAD / 2, W - PAD, H - PAD)

    # Header: service letter box + carrier and service.
    top = H - PAD
    c.setFillColor(black)
    c.rect(PAD, top - 0.75 * inch, 0.75 * inch, 0.75 * inch, fill=1)
    c.setFillColor(white)
    c.setFont("Helvetica-Bold", 40)
    letter = (service[:1] or "S").upper()
    c.drawCentredString(PAD + 0.375 * inch, top - 0.6 * inch, letter)
    c.setFillColor(black)
    head_x = PAD + 0.9 * inch
    head_w = W - head_x - PAD
    carrier_txt = carrier.upper()
    c.setFont("Helvetica-Bold", _fit(c, carrier_txt, "Helvetica-Bold", 18, head_w))
    c.drawString(head_x, top - 0.3 * inch, carrier_txt)
    service_txt = service.upper()
    c.setFont("Helvetica-Bold", _fit(c, service_txt, "Helvetica-Bold", 13, head_w))
    c.drawString(head_x, top - 0.55 * inch, service_txt)
    c.setFont("Helvetica", 8)
    c.drawString(head_x, top - 0.72 * inch, f"{weight_oz:g} OZ  ·  {weight_oz / 16:.2f} LB")
    y = top - 0.88 * inch
    c.line(PAD, y, W - PAD, y)

    # From
    c.setFont("Helvetica-Bold", 7)
    c.drawString(PAD, y - 0.16 * inch, "FROM")
    c.setFont("Helvetica", 9)
    c.drawString(PAD + 0.45 * inch, y - 0.16 * inch, sender.name[:48])
    c.drawString(
        PAD + 0.45 * inch, y - 0.31 * inch, f"{sender.city}, {sender.state} {sender.zip}"[:48]
    )
    y -= 0.45 * inch
    c.line(PAD, y, W - PAD, y)

    # To
    c.setFont("Helvetica-Bold", 8)
    c.drawString(PAD, y - 0.2 * inch, "SHIP TO")
    name_size = _fit(c, recipient.name.upper(), "Helvetica-Bold", 16, inner - 0.3 * inch)
    c.setFont("Helvetica-Bold", name_size)
    c.drawString(PAD + 0.3 * inch, y - 0.5 * inch, recipient.name.upper())
    city = f"{recipient.city}, {recipient.state} {recipient.zip}".upper()
    c.setFont("Helvetica-Bold", _fit(c, city, "Helvetica-Bold", 16, inner - 0.3 * inch))
    c.drawString(PAD + 0.3 * inch, y - 0.78 * inch, city)
    y -= 1.0 * inch
    c.setLineWidth(3)
    c.line(PAD, y, W - PAD, y)
    c.setLineWidth(1.5)

    # Tracking barcode
    c.setFont("Helvetica-Bold", 9)
    c.drawCentredString(W / 2, y - 0.2 * inch, "TRACKING #")
    bar_h = 1.0 * inch
    bc = code128.Code128(tracking_code, barHeight=bar_h, barWidth=1.0, quiet=False)
    if bc.width > inner:
        bc = code128.Code128(tracking_code, barHeight=bar_h, barWidth=inner / bc.width, quiet=False)
    bc.drawOn(c, (W - bc.width) / 2, y - 0.3 * inch - bar_h)
    c.setFont("Courier-Bold", _fit(c, tracking_code, "Courier-Bold", 12, inner))
    c.drawCentredString(W / 2, y - 0.5 * inch - bar_h, tracking_code)
    y -= 0.62 * inch + bar_h
    c.line(PAD, y, W - PAD, y)

    c.setFont("Helvetica", 7)
    c.drawString(PAD, PAD, f"Shipment {shipment_id}"[:70])
    c.drawRightString(W - PAD, PAD, "InvAI sandbox carrier")

    # Watermark
    c.saveState()
    c.setFillColor(Color(0.86, 0.15, 0.15, alpha=0.28))
    c.translate(W / 2, H / 2)
    c.rotate(55)
    c.setFont("Helvetica-Bold", 30)
    c.drawCentredString(0, 14, "SAMPLE —")
    c.drawCentredString(0, -22, "NOT A REAL LABEL")
    c.restoreState()
    c.setFillColor(Color(0.86, 0.15, 0.15))
    c.setFont("Helvetica-Bold", 8)
    c.drawCentredString(W / 2, y - 0.2 * inch, "SAMPLE — NOT A REAL LABEL — DO NOT SHIP")

    c.showPage()
    c.save()
    return buf.getvalue()
