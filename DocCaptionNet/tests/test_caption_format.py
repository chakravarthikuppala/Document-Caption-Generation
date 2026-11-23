# DocCaptionNet
from doccaptionnet.caption_format import format_caption


def test_doccaptionnet_includes_document_name() -> None:
    caption = format_caption("DocCaptionNet_invoice.png", "an invoice with a total due")
    assert caption.startswith("[Document: DocCaptionNet_invoice.png]")
    assert "invoice" in caption


def test_doccaptionnet_page_prefix() -> None:
    caption = format_caption("report.pdf", "a table of figures", page=2, total_pages=5)
    assert caption == "[Document: report.pdf | Page 2/5] a table of figures"
