# DocCaptionNet
from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from doccaptionnet.caption_format import document_name, format_caption
from doccaptionnet.config import ROOT
from doccaptionnet.preprocess import prepare_image


class DocCaptionNetDataset(Dataset):
    def __init__(
        self,
        csv_path: str | Path,
        processor,
        max_target_length: int = 64,
        include_document_name: bool = True,
    ) -> None:
        self.df = pd.read_csv(csv_path)
        if "image_path" not in self.df.columns or "caption" not in self.df.columns:
            raise ValueError("DocCaptionNet CSV must have image_path and caption columns")
        self.processor = processor
        self.max_target_length = max_target_length
        self.include_document_name = include_document_name

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> dict:
        row = self.df.iloc[index]
        image_path = Path(str(row["image_path"]))
        if not image_path.is_absolute():
            image_path = ROOT / image_path
        image = prepare_image(Image.open(image_path).convert("RGB"))
        caption = str(row["caption"])
        if self.include_document_name:
            caption = format_caption(document_name(image_path), caption)
        encoding = self.processor(
            images=image,
            text=caption,
            padding="max_length",
            truncation=True,
            max_length=self.max_target_length,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in encoding.items()}
        item["labels"] = item["input_ids"].clone()
        return item
