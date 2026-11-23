# DocCaptionNet
"""Always prefix generated captions with the source document name."""

from __future__ import annotations

from pathlib import Path


def document_name(path: str | Path) -> str:
    return Path(path).name


def format_caption(
    document_name_value: str,
    text: str,
    page: int | None = None,
    total_pages: int | None = None,
) -> str:
    cleaned = " ".join((text or "").strip().split())
    if not cleaned:
        cleaned = "No caption was generated."
    if page is None:
        prefix = f"[Document: {document_name_value}]"
    else:
        total = total_pages if total_pages is not None else page
        prefix = f"[Document: {document_name_value} | Page {page}/{total}]"
    if cleaned.startswith("[Document:"):
        return cleaned
    return f"{prefix} {cleaned}"
