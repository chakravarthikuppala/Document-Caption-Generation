# DocCaptionNet
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd


def _page(path: Path, title: str, body: str, color: tuple[int, int, int]) -> None:
    image = Image.new("RGB", (768, 1024), color)
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 36)
        small = ImageFont.truetype("arial.ttf", 24)
    except OSError:
        font = ImageFont.load_default()
        small = font
    draw.rectangle((48, 48, 720, 976), outline=(15, 61, 92), width=4)
    draw.text((72, 80), title, fill=(15, 61, 92), font=font)
    draw.text((72, 160), body, fill=(40, 70, 90), font=small)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def main() -> None:
    raw = ROOT / "data" / "raw"
    rows = [
        ("DocCaptionNet_invoice.png", "INVOICE #4412", "Acme Supplies — total due $1,240", (244, 239, 230), "an invoice listing line items and a payment total"),
        ("DocCaptionNet_letter.png", "OFFICIAL LETTER", "Dear committee, please find enclosed…", (232, 238, 243), "a formal letter on letterhead"),
        ("DocCaptionNet_report.png", "QUARTERLY REPORT", "Revenue grew 12% versus last quarter.", (236, 232, 222), "a quarterly business report with a summary heading"),
    ]
    records = []
    for name, title, body, color, caption in rows:
        path = raw / name
        _page(path, title, body, color)
        records.append({"image_path": path.relative_to(ROOT).as_posix(), "caption": caption})
    processed = ROOT / "data" / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    df.to_csv(processed / "DocCaptionNet_train.csv", index=False)
    df.to_csv(processed / "DocCaptionNet_val.csv", index=False)
    print("DocCaptionNet wrote sample images and CSVs.")


if __name__ == "__main__":
    main()
