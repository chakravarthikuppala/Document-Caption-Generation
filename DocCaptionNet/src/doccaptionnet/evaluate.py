# DocCaptionNet
from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image

from doccaptionnet.caption_format import document_name
from doccaptionnet.config import resolve_path
from doccaptionnet.infer import caption_pages
from doccaptionnet.model import DocCaptionNetModel
from doccaptionnet.document_io import DocumentPage


def evaluate_csv(csv_path: str | Path, model: DocCaptionNetModel | None = None) -> dict:
    df = pd.read_csv(resolve_path(csv_path))
    predictions: list[str] = []
    references: list[str] = []
    model = model or DocCaptionNetModel()
    for _, row in df.iterrows():
        image_path = resolve_path(str(row["image_path"]))
        image = Image.open(image_path).convert("RGB")
        pages = [
            DocumentPage(
                document_name=document_name(image_path),
                page=1,
                total_pages=1,
                image=image,
                source_path=str(image_path),
            )
        ]
        rec = caption_pages(pages, model=model)[0]
        predictions.append(rec["caption"])
        references.append(str(row["caption"]))
    overlap = [
        int(any(tok in pred.lower() for tok in ref.lower().split()[:4]))
        for pred, ref in zip(predictions, references)
    ]
    return {
        "project": "DocCaptionNet",
        "n": len(predictions),
        "token_hit_rate": sum(overlap) / max(1, len(overlap)),
        "predictions": predictions,
        "references": references,
    }
