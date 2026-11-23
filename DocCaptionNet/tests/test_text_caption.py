# DocCaptionNet
from PIL import Image

from doccaptionnet.document_io import DocumentPage
from doccaptionnet.infer import caption_pages
from doccaptionnet.model import DocCaptionNetModel
from doccaptionnet.text_caption import caption_from_text, compose_raw_caption


def test_caption_from_text_uses_content() -> None:
    caption = caption_from_text(
        "Abstract\nBugLens AI Mini Project Report\nThis work detects software bugs with a CNN."
    )
    assert "abstract" in caption.lower()
    assert "BugLens" in caption or "bugs" in caption.lower()


def test_pages_get_different_captions() -> None:
    blank = Image.new("RGB", (200, 280), (250, 248, 240))
    pages = [
        DocumentPage(
            document_name="BugLens_AI_Mini_Project_Report(final).pdf",
            page=1,
            total_pages=2,
            image=blank.copy(),
            page_text="Abstract This report introduces BugLens, a mini project for AI bug detection.",
        ),
        DocumentPage(
            document_name="BugLens_AI_Mini_Project_Report(final).pdf",
            page=2,
            total_pages=2,
            image=blank.copy(),
            page_text="References\nSmith, J. 2020. Neural bug finding.\nJones, A. 2021. Static analysis.",
        ),
    ]
    records = caption_pages(pages, model=DocCaptionNetModel(force_heuristic=True), visual_mode="auto")
    assert records[0]["raw_caption"] != records[1]["raw_caption"]
    assert "Abstract" in records[0]["caption"] or "abstract" in records[0]["caption"]
    assert "reference" in records[1]["caption"].lower()
    assert records[0]["caption"].startswith("[Document: BugLens_AI_Mini_Project_Report(final).pdf | Page 1/2]")


def test_compose_prefers_page_text_over_generic_visual() -> None:
    composed = compose_raw_caption(
        visual="a portrait high-contrast document page (543x768px) with printed layout suitable for captioning",
        page_text="Introduction The system architecture uses a two-stage detector.",
        backend="heuristic",
    )
    assert "suitable for captioning" not in composed
    assert "Introduction" in composed or "architecture" in composed
