# DocCaptionNet
"""python -m doccaptionnet — run inference or the API."""

from __future__ import annotations

import argparse
from pathlib import Path

from doccaptionnet.infer import caption_path, save_history
from doccaptionnet.logging_utils import get_logger

log = get_logger("DocCaptionNet")


def main() -> None:
    parser = argparse.ArgumentParser(prog="DocCaptionNet")
    parser.add_argument("path", nargs="?", help="Image or PDF path")
    parser.add_argument("--api", action="store_true", help="Start FastAPI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    if args.api:
        import uvicorn

        uvicorn.run("doccaptionnet.api:app", host=args.host, port=args.port, reload=False)
        return

    if not args.path:
        parser.error("Provide a document path or --api")

    records = caption_path(Path(args.path))
    for rec in records:
        print(rec["caption"])
    save_history(records)
    log.info("DocCaptionNet wrote %s caption(s)", len(records))


if __name__ == "__main__":
    main()
