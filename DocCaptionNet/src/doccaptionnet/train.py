# DocCaptionNet
from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BlipForConditionalGeneration, BlipProcessor

from doccaptionnet.config import TrainConfig, load_train_config, resolve_path
from doccaptionnet.dataset import DocCaptionNetDataset
from doccaptionnet.logging_utils import get_logger

log = get_logger("DocCaptionNet")


def train_doccaptionnet(cfg: TrainConfig | None = None) -> Path:
    cfg = cfg or load_train_config()
    torch.manual_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = BlipProcessor.from_pretrained(cfg.model_id)
    model = BlipForConditionalGeneration.from_pretrained(cfg.model_id).to(device)
    train_ds = DocCaptionNetDataset(
        resolve_path(cfg.train_csv),
        processor,
        cfg.max_target_length,
        cfg.include_document_name,
    )
    loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    model.train()
    for epoch in range(cfg.epochs):
        running = 0.0
        for batch in tqdm(loader, desc=f"DocCaptionNet epoch {epoch + 1}/{cfg.epochs}"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(
                pixel_values=batch["pixel_values"],
                input_ids=batch["input_ids"],
                attention_mask=batch.get("attention_mask"),
                labels=batch["labels"],
            )
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += float(loss.item())
        log.info("DocCaptionNet epoch %s loss %.4f", epoch + 1, running / max(1, len(loader)))
    output_dir = resolve_path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    log.info("Saved DocCaptionNet checkpoint to %s", output_dir)
    return output_dir
