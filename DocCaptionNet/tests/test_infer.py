# DocCaptionNet
from io import BytesIO

from PIL import Image

from doccaptionnet.infer import caption_upload
from doccaptionnet.model import DocCaptionNetModel


def test_doccaptionnet_end_to_end_heuristic() -> None:
    buffer = BytesIO()
    Image.new("RGB", (64, 80), (250, 248, 240)).save(buffer, format="PNG")
    model = DocCaptionNetModel(force_heuristic=True)
    records = caption_upload("DocCaptionNet_memo.png", buffer.getvalue(), model=model)
    assert records[0]["document_name"] == "DocCaptionNet_memo.png"
    assert records[0]["caption"].startswith("[Document: DocCaptionNet_memo.png]")
    assert "Page " not in records[0]["caption"]
    assert records[0]["backend"] == "heuristic"
