# DocCaptionNet — all files

Every path below belongs to the **DocCaptionNet** project. Generated captions always include the source document name via `src/doccaptionnet/caption_format.py`.

| File | Role |
| --- | --- |
| `README.md` | DocCaptionNet setup and usage |
| `LICENSE` | MIT license for DocCaptionNet |
| `.gitignore` | Ignore weights, venv, generated captions |
| `.env.example` | DocCaptionNet environment variables |
| `requirements.txt` | Python dependencies |
| `environment.yml` | Conda environment |
| `pyproject.toml` | Package metadata (`doccaptionnet`) |
| `Dockerfile` | Container for UI |
| `docker-compose.yml` | UI + API services |
| `FILE_MANIFEST.md` | This index |
| `configs/DocCaptionNet_inference.yaml` | Inference settings |
| `configs/DocCaptionNet_train.yaml` | Training settings |
| `data/raw/.gitkeep` | Raw document images |
| `data/processed/.gitkeep` | Train/val CSV placeholders |
| `data/samples/README.md` | How to add demo documents |
| `notebooks/01_DocCaptionNet_eda.ipynb` | Sample-page notebook |
| `src/doccaptionnet/__init__.py` | Package version |
| `src/doccaptionnet/__main__.py` | CLI entry |
| `src/doccaptionnet/config.py` | YAML + env config |
| `src/doccaptionnet/logging_utils.py` | Logger |
| `src/doccaptionnet/document_io.py` | Image/PDF load, keep filename |
| `src/doccaptionnet/preprocess.py` | Resize/RGB |
| `src/doccaptionnet/caption_format.py` | Inject document name into captions |
| `src/doccaptionnet/text_caption.py` | Unique captions from extracted PDF text |
| `src/doccaptionnet/model.py` | BLIP + heuristic backends |
| `src/doccaptionnet/dataset.py` | Fine-tune dataset |
| `src/doccaptionnet/train.py` | Training loop |
| `src/doccaptionnet/evaluate.py` | Simple eval |
| `src/doccaptionnet/infer.py` | Caption + history |
| `src/doccaptionnet/api.py` | FastAPI app |
| `.streamlit/config.toml` | Light theme so captions stay readable |
| `app/DocCaptionNet_streamlit.py` | Streamlit UI |
| `app/components.py` | Shared UI chrome |
| `app/assets/DocCaptionNet_logo.svg` | Logo |
| `scripts/download_pretrained.py` | Cache BLIP weights |
| `scripts/prepare_dataset.py` | Synthetic DocCaptionNet pages |
| `scripts/train_doccaptionnet.py` | Train entrypoint |
| `scripts/run_batch.py` | Folder batch captions |
| `tests/test_caption_format.py` | Name-prefix tests |
| `tests/test_document_io.py` | Upload naming tests |
| `tests/test_infer.py` | End-to-end heuristic test |
| `tests/test_text_caption.py` | Per-page unique caption tests |
| `models/.gitkeep` | Checkpoint directory |
| `outputs/captions/.gitkeep` | Caption export directory |
