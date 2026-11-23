# DocCaptionNet
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from collections.abc import Callable

from doccaptionnet.caption_format import format_caption
from doccaptionnet.config import load_inference_config, resolve_path
from doccaptionnet.document_io import DocumentPage, load_document, load_upload
from doccaptionnet.model import DocCaptionNetModel, get_model
from doccaptionnet.text_caption import caption_from_text, compose_raw_caption


ProgressFn = Callable[[int, int, dict], None]


def caption_pages(
    pages: Iterable[DocumentPage],
    model: DocCaptionNetModel | None = None,
    on_progress: ProgressFn | None = None,
    visual_mode: str = "auto",
    **_unused,
) -> list[dict]:
    cfg = load_inference_config()
    model = model or get_model()
    page_list = list(pages)
    records: list[dict] = []
    total = len(page_list)
    for page in page_list:
        has_text = bool(caption_from_text(page.page_text))
        if visual_mode == "text":
            run_visual = False
        elif model.backend_name == "blip":
            run_visual = visual_mode == "blip" or not has_text
        else:
            run_visual = not has_text
        visual = model.caption_image(page.image) if run_visual else ""
        raw = compose_raw_caption(
            visual=visual,
            page_text=page.page_text,
            backend=model.backend_name if visual else "text",
        )
        page_no = None if page.total_pages <= 1 else page.page
        total_pages = None if page.total_pages <= 1 else page.total_pages
        caption = (
            format_caption(page.document_name, raw, page_no, total_pages)
            if cfg.include_document_name
            else raw
        )
        source = "blip" if model.backend_name == "blip" and visual else "page-text"
        if not (page.page_text or "").strip() and model.backend_name != "blip":
            source = "heuristic"
        elif (page.page_text or "").strip() and model.backend_name == "blip" and visual:
            source = "blip+page-text"
        record = {
            "project": "DocCaptionNet",
            "document_name": page.document_name,
            "page": page.page,
            "total_pages": page.total_pages,
            "caption": caption,
            "raw_caption": raw,
            "backend": source,
            "model_id": model.cfg.model_id,
            "source_path": page.source_path,
            "page_text_preview": " ".join((page.page_text or "").split())[:240],
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        records.append(record)
        if on_progress:
            on_progress(len(records), total, record)
    return records


def caption_path(
    path: str | Path,
    model: DocCaptionNetModel | None = None,
    visual_mode: str = "auto",
) -> list[dict]:
    return caption_pages(load_document(path), model=model, visual_mode=visual_mode)


def caption_upload(
    name: str,
    raw: bytes,
    model: DocCaptionNetModel | None = None,
    visual_mode: str = "auto",
    on_progress: ProgressFn | None = None,
) -> list[dict]:
    return caption_pages(
        load_upload(name, raw),
        model=model,
        visual_mode=visual_mode,
        on_progress=on_progress,
    )


def save_history(records: list[dict], history_path: str | Path | None = None) -> Path:
    cfg = load_inference_config()
    path = resolve_path(history_path or cfg.history_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return path


def load_history(history_path: str | Path | None = None, limit: int = 200) -> list[dict]:
    cfg = load_inference_config()
    path = resolve_path(history_path or cfg.history_path)
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines if line.strip()]
    return rows[-limit:]
