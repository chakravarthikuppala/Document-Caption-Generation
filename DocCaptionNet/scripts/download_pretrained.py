# DocCaptionNet
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from transformers import BlipForConditionalGeneration, BlipProcessor

from doccaptionnet.config import load_inference_config, resolve_path


def main() -> None:
    cfg = load_inference_config()
    dest = resolve_path("models/DocCaptionNet_blip_base")
    dest.mkdir(parents=True, exist_ok=True)
    print(f"DocCaptionNet downloading {cfg.model_id} -> {dest}")
    processor = BlipProcessor.from_pretrained(cfg.model_id)
    model = BlipForConditionalGeneration.from_pretrained(cfg.model_id)
    processor.save_pretrained(dest)
    model.save_pretrained(dest)
    print("Done.")


if __name__ == "__main__":
    main()
