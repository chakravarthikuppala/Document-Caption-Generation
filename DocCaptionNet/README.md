# DocCaptionNet

End-to-end **document caption generation** with a vision-language model, FastAPI, and a Streamlit UI. Every caption includes the **document file name** (and page number for PDFs).

`FILE_MANIFEST.md` lists every file in this repository.

## What it does

1. You upload a PNG/JPEG/WebP/TIFF or PDF.
2. DocCaptionNet renders each page to an image **and extracts that page’s text**.
3. Auto mode writes a **unique caption from the page content**. BLIP is used for scans with no text (requires `torch` + `transformers`).
4. The caption is stored as:

```text
[Document: invoice.pdf | Page 2/5] a scanned invoice listing line items and a total due
```

## Quick start (Windows)

```powershell
cd C:\Users\mokak\DocCaptionNet
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
python scripts/prepare_dataset.py
streamlit run app/DocCaptionNet_streamlit.py
```

Open the URL Streamlit prints (usually `http://localhost:8501`).

Leave the sidebar on **Auto**. That reads each PDF page so captions are not identical. Choose **BLIP neural captions** after installing PyTorch if you want vision-language captions for scanned pages.

### API

```powershell
$env:PYTHONPATH = "src"
uvicorn doccaptionnet.api:app --reload
```

- `GET /health`
- `POST /caption` (multipart file)
- `GET /history`
- `GET /files` (this repo’s `FILE_MANIFEST.md`)

### CLI

```powershell
$env:PYTHONPATH = "src"
python -m doccaptionnet data\raw\DocCaptionNet_invoice.png
```

## Training

```powershell
python scripts/prepare_dataset.py
python scripts/train_doccaptionnet.py
```

Point `configs/DocCaptionNet_inference.yaml` `checkpoint` at `models/DocCaptionNet_finetuned` after training.

## Tests

```powershell
pytest
```

## Project layout

See `FILE_MANIFEST.md` for the full named file list.
