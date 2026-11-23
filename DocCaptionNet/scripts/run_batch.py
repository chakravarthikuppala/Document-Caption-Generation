# DocCaptionNet
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from doccaptionnet.infer import caption_path, save_history


def main() -> None:
    parser = argparse.ArgumentParser(prog="DocCaptionNet batch")
    parser.add_argument("folder", type=Path)
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "captions" / "DocCaptionNet_batch.csv")
    args = parser.parse_args()
    records: list[dict] = []
    for path in sorted(args.folder.iterdir()):
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".pdf"}:
            records.extend(caption_path(path))
    save_history(records)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(args.out, index=False)
    print(f"DocCaptionNet wrote {args.out}")


if __name__ == "__main__":
    main()
