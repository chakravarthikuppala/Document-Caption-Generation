# DocCaptionNet
from __future__ import annotations

from typing import Protocol

from PIL import Image

from doccaptionnet.config import InferenceConfig, load_inference_config
from doccaptionnet.logging_utils import get_logger
from doccaptionnet.preprocess import prepare_image

log = get_logger("DocCaptionNet")


class CaptionBackend(Protocol):
    def generate(self, image: Image.Image, max_new_tokens: int, num_beams: int) -> str: ...


class HeuristicCaptioner:
    """Last-resort visual fallback when BLIP cannot load and the page has no text."""

    def generate(self, image: Image.Image, max_new_tokens: int, num_beams: int) -> str:
        gray = image.convert("L")
        gray.thumbnail((160, 220))
        width, height = gray.size
        hist = gray.histogram()
        total = max(1, sum(hist))
        ink = sum(hist[:110]) / total
        extrema = gray.getextrema()
        tone = "high-contrast" if extrema[1] - extrema[0] > 80 else "low-contrast"
        orientation = "portrait" if height >= width else "landscape"
        if ink > 0.18:
            layout = "dense printed text or diagrams"
        elif ink > 0.05:
            layout = "mixed text and whitespace"
        else:
            layout = "a mostly blank or lightly printed sheet"
        caption = (
            f"a {orientation} {tone} document page "
            f"({width}x{height}px) with {layout} (ink {ink:.0%})"
        )
        return caption[: max(24, max_new_tokens * 8)]


class BlipCaptioner:
    def __init__(self, model_id: str, checkpoint: str, device: str) -> None:
        import torch
        from transformers import BlipForConditionalGeneration, BlipProcessor

        self.device = _resolve_device(device)
        source = checkpoint or model_id
        log.info("Loading DocCaptionNet BLIP weights from %s on %s", source, self.device)
        self.processor = BlipProcessor.from_pretrained(source)
        self.model = BlipForConditionalGeneration.from_pretrained(source)
        self.model.to(self.device)
        self.model.eval()
        self.torch = torch

    def generate(self, image: Image.Image, max_new_tokens: int, num_beams: int) -> str:
        prompt = "a document page showing"
        inputs = self.processor(images=image, text=prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with self.torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
            )
        text = self.processor.decode(output_ids[0], skip_special_tokens=True)
        return " ".join(text.split())


def _resolve_device(device: str) -> str:
    if device and device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


class DocCaptionNetModel:
    load_error: str | None = None

    def __init__(self, cfg: InferenceConfig | None = None, force_heuristic: bool = False) -> None:
        self.cfg = cfg or load_inference_config()
        self.backend: CaptionBackend
        self.load_error = None
        if force_heuristic:
            self.backend = HeuristicCaptioner()
            self.backend_name = "heuristic"
            return
        try:
            self.backend = BlipCaptioner(self.cfg.model_id, self.cfg.checkpoint, self.cfg.device)
            self.backend_name = "blip"
        except Exception as exc:  # pragma: no cover - environment dependent
            self.load_error = str(exc)
            log.warning("BLIP unavailable (%s); DocCaptionNet will caption from page text", exc)
            self.backend = HeuristicCaptioner()
            self.backend_name = "heuristic"

    def caption_image(self, image: Image.Image) -> str:
        prepared = prepare_image(image)
        return self.backend.generate(
            prepared,
            max_new_tokens=self.cfg.max_new_tokens,
            num_beams=self.cfg.num_beams,
        )


_MODEL: DocCaptionNetModel | None = None


def get_model(force_heuristic: bool = False) -> DocCaptionNetModel:
    global _MODEL
    if _MODEL is None or (force_heuristic and _MODEL.backend_name != "heuristic"):
        _MODEL = DocCaptionNetModel(force_heuristic=force_heuristic)
    return _MODEL
