# DocCaptionNet
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from doccaptionnet import PROJECT_NAME, __version__
from doccaptionnet.config import ROOT
from doccaptionnet.infer import caption_upload, load_history, save_history

app = FastAPI(title="DocCaptionNet", version=__version__)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {"project": PROJECT_NAME, "status": "ok", "version": __version__}


@app.post("/caption")
async def caption(file: UploadFile = File(...)) -> dict:
    if not file.filename:
        raise HTTPException(400, "DocCaptionNet requires a file name")
    raw = await file.read()
    try:
        records = caption_upload(file.filename, raw)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    save_history(records)
    return {"project": PROJECT_NAME, "document_name": file.filename, "captions": records}


@app.get("/history")
def history() -> dict:
    return {"project": PROJECT_NAME, "items": load_history()}


@app.get("/files")
def files() -> FileResponse:
    manifest = ROOT / "FILE_MANIFEST.md"
    return FileResponse(manifest, media_type="text/markdown", filename="DocCaptionNet_FILE_MANIFEST.md")
