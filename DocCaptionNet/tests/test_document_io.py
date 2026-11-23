# DocCaptionNet
from io import BytesIO

from PIL import Image

from doccaptionnet.document_io import load_upload


def test_doccaptionnet_image_upload_keeps_name() -> None:
    buffer = BytesIO()
    Image.new("RGB", (32, 40), (240, 240, 240)).save(buffer, format="PNG")
    pages = load_upload("DocCaptionNet_scan.png", buffer.getvalue())
    assert len(pages) == 1
    assert pages[0].document_name == "DocCaptionNet_scan.png"
    assert pages[0].page == 1
