# DocCaptionNet
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INFERENCE = ROOT / "configs" / "DocCaptionNet_inference.yaml"
DEFAULT_TRAIN = ROOT / "configs" / "DocCaptionNet_train.yaml"


@dataclass
class InferenceConfig:
    project: str = "DocCaptionNet"
    model_id: str = "Salesforce/blip-image-captioning-base"
    checkpoint: str = ""
    device: str = "auto"
    max_new_tokens: int = 40
    num_beams: int = 3
    history_path: str = "outputs/history.jsonl"
    include_document_name: bool = True


@dataclass
class TrainConfig:
    project: str = "DocCaptionNet"
    model_id: str = "Salesforce/blip-image-captioning-base"
    output_dir: str = "models/DocCaptionNet_finetuned"
    train_csv: str = "data/processed/DocCaptionNet_train.csv"
    val_csv: str = "data/processed/DocCaptionNet_val.csv"
    epochs: int = 3
    batch_size: int = 2
    learning_rate: float = 5e-5
    max_target_length: int = 64
    include_document_name: bool = True
    seed: int = 42


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_inference_config(path: Path | None = None) -> InferenceConfig:
    data = _load_yaml(path or DEFAULT_INFERENCE)
    cfg = InferenceConfig(**{k: v for k, v in data.items() if k in InferenceConfig.__dataclass_fields__})
    cfg.model_id = os.getenv("DOCCAPTIONNET_MODEL_ID", cfg.model_id)
    cfg.device = os.getenv("DOCCAPTIONNET_DEVICE", cfg.device)
    cfg.checkpoint = os.getenv("DOCCAPTIONNET_CHECKPOINT", cfg.checkpoint)
    if os.getenv("DOCCAPTIONNET_MAX_NEW_TOKENS"):
        cfg.max_new_tokens = int(os.environ["DOCCAPTIONNET_MAX_NEW_TOKENS"])
    if os.getenv("DOCCAPTIONNET_NUM_BEAMS"):
        cfg.num_beams = int(os.environ["DOCCAPTIONNET_NUM_BEAMS"])
    cfg.history_path = os.getenv("DOCCAPTIONNET_HISTORY", cfg.history_path)
    return cfg


def load_train_config(path: Path | None = None) -> TrainConfig:
    data = _load_yaml(path or DEFAULT_TRAIN)
    return TrainConfig(**{k: v for k, v in data.items() if k in TrainConfig.__dataclass_fields__})


def resolve_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p
