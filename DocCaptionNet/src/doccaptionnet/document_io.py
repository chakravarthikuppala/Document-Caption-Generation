# DocCaptionNet
from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from PIL import Image

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
PDF_EXTS = {".pdf"}


@dataclass
class DocumentPage:
    document_name: str
    page: int
    total_pages: int
    image: Image.Image
    source_path: str | None = None
    page_text: str = ""


def _open_image(data: BytesIO | Path) -> Image.Image:
    image = Image.open(data)
    return image.convert("RGB")


def load_image_bytes(name: str, raw: bytes) -> list[DocumentPage]:
    image = _open_image(BytesIO(raw))
    return [
        DocumentPage(
            document_name=name,
            page=1,
            total_pages=1,
            image=image,
        )
    ]


def _pdf_page_text(page) -> str:
    textpage = page.get_textpage()
    try:
        if hasattr(textpage, "get_text_range"):
            extracted = textpage.get_text_range() or ""
        else:
            extracted = textpage.get_text_bounded() or ""
    finally:
        textpage.close()
    return extracted.strip()


def load_pdf_bytes(name: str, raw: bytes, scale: float = 0.7) -> list[DocumentPage]:
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(raw)
    pages: list[DocumentPage] = []
    total = len(pdf)
    for index in range(total):
        page = pdf[index]
        text = _pdf_page_text(page)
        bitmap = page.render(scale=scale)
        image = bitmap.to_pil().convert("RGB")
        pages.append(
            DocumentPage(
                document_name=name,
                page=index + 1,
                total_pages=total,
                image=image,
                page_text=text,
            )
        )
    pdf.close()
    return pages


def load_document(path: str | Path) -> list[DocumentPage]:
    path = Path(path)
    suffix = path.suffix.lower()
    raw = path.read_bytes()
    if suffix in IMAGE_EXTS:
        pages = load_image_bytes(path.name, raw)
    elif suffix in PDF_EXTS:
        pages = load_pdf_bytes(path.name, raw)
    else:
        raise ValueError(f"DocCaptionNet does not support {suffix} files.")
    for page in pages:
        page.source_path = str(path)
    return pages


def load_upload(name: str, raw: bytes) -> list[DocumentPage]:
    suffix = Path(name).suffix.lower()
    if suffix in IMAGE_EXTS:
        return load_image_bytes(name, raw)
    if suffix in PDF_EXTS:
        return load_pdf_bytes(name, raw)
    raise ValueError(f"DocCaptionNet does not support {suffix} files.")
