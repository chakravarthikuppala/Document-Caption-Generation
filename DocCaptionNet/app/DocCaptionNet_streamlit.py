# DocCaptionNet Streamlit UI
from __future__ import annotations

import html
import sys
from io import BytesIO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd
import streamlit as st
from PIL import Image

from components import hero, inject_style, records_table
from doccaptionnet.document_io import load_upload
from doccaptionnet.infer import caption_pages, load_history, save_history
from doccaptionnet.model import DocCaptionNetModel

st.set_page_config(page_title="DocCaptionNet", page_icon="📄", layout="wide")
inject_style()
hero()

ENGINE_TO_MODE = {
    "Auto — unique page text (recommended)": "auto",
    "BLIP neural captions": "blip",
    "Page text only": "text",
}
MODEL_CACHE_VERSION = 4


@st.cache_resource(show_spinner="Loading DocCaptionNet captioner…")
def load_caption_model(want_blip: bool, cache_version: int) -> DocCaptionNetModel:
    return DocCaptionNetModel(force_heuristic=not want_blip)


def thumb(image: Image.Image, size: tuple[int, int] = (200, 280)) -> Image.Image:
    preview = image.copy()
    preview.thumbnail(size, Image.Resampling.BILINEAR)
    return preview


with st.sidebar:
    st.subheader("DocCaptionNet settings")
    engine = st.radio(
        "Caption engine",
        list(ENGINE_TO_MODE),
        index=0,
        help="Auto is fast and unique per page. Choose BLIP only for scanned pages with no selectable text.",
    )
    visual_mode = ENGINE_TO_MODE[engine]
    want_blip = visual_mode == "blip"
    model = load_caption_model(want_blip, MODEL_CACHE_VERSION)
    if visual_mode == "blip" and model.backend_name == "blip":
        st.success("BLIP is ready.")
    elif visual_mode == "blip":
        st.error(
            "BLIP could not start. Auto still captions from page text. "
            "From the project folder run:  pip install torch transformers"
        )
        err = getattr(model, "load_error", None)
        if err:
            st.caption(err)
    else:
        st.info("Fast page-text captions. BLIP is not required for digital PDFs.")
    st.caption(f"Engine in use: **{model.backend_name if want_blip else 'page-text'}**")
    page = st.radio(
        "Workspace",
        ["Single document", "Batch", "History", "About & files"],
        index=0,
    )


def render_records(records: list[dict], pages=None) -> None:
    thumbs = [p.image for p in pages] if pages is not None else [None] * len(records)
    for rec, image in zip(records, thumbs):
        with st.expander(f"Page {rec['page']} / {rec['total_pages']}", expanded=rec["page"] <= 3):
            if image is not None:
                left, right = st.columns([1, 3])
                with left:
                    st.image(thumb(image), use_container_width=True)
                body = right
            else:
                body = st.container()
            with body:
                caption = html.escape(rec["caption"])
                backend = html.escape(str(rec["backend"]))
                st.markdown(
                    f"<div class='dcn-card'><div class='dcn-caption'>{caption}</div>"
                    f"<div class='dcn-meta'>Engine: {backend}</div></div>",
                    unsafe_allow_html=True,
                )
    st.download_button(
        "Download captions CSV",
        pd.DataFrame(records).to_csv(index=False).encode("utf-8"),
        file_name=f"DocCaptionNet_{Path(records[0]['document_name']).stem}.csv",
        mime="text/csv",
        key=f"dl-{records[0]['document_name']}-{len(records)}",
    )


if page == "Single document":
    st.markdown(
        """
        <div class="dcn-section">
          <h2>Create a document caption</h2>
          <p>Upload an image or PDF to generate a caption for every page.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    uploaded = st.file_uploader(
        "Choose a document",
        type=["png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff", "pdf"],
        help="Supported formats: PNG, JPG, WEBP, BMP, TIFF, and PDF.",
    )
    if uploaded:
        raw = uploaded.getvalue()
        is_pdf = Path(uploaded.name).suffix.lower() == ".pdf"
        with st.container(border=True):
            preview_column, details_column = st.columns([0.85, 1.15], vertical_alignment="center")
            with preview_column:
                if not is_pdf:
                    st.image(
                        Image.open(BytesIO(raw)),
                        caption="Document preview",
                        use_container_width=True,
                    )
                else:
                    st.info("PDF document\n\nEach page will be processed separately.")
            with details_column:
                st.markdown("#### Document ready")
                st.write(uploaded.name)
                st.caption(f"{'PDF' if is_pdf else 'Image'} · {len(raw) / 1024:.1f} KB")
        if st.button("Generate caption", type="primary"):
            progress = st.progress(0.0)
            status = st.empty()

            def on_progress(done: int, total: int, rec: dict) -> None:
                progress.progress(done / max(1, total))
                status.write(f"Captioned page {rec['page']} / {rec['total_pages']}")

            with st.spinner("Reading pages…"):
                pages = load_upload(uploaded.name, raw)
                records = caption_pages(
                    pages,
                    model=model,
                    visual_mode=visual_mode,
                    on_progress=on_progress,
                )
                save_history(records)
            unique = {r["raw_caption"] for r in records}
            if len(records) > 1 and len(unique) == 1:
                st.error("Every page produced the same caption. The file may be a scan with no extractable text.")
            else:
                st.success(f"Generated {len(records)} caption(s) for `{uploaded.name}`.")
            render_records(records, pages)

elif page == "Batch":
    st.markdown(
        """
        <div class="dcn-section">
          <h2>Batch captioning</h2>
          <p>Process multiple documents together and download the results as a CSV.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    files = st.file_uploader(
        "Choose documents",
        type=["png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff", "pdf"],
        accept_multiple_files=True,
    )
    if files and st.button("Caption all", type="primary"):
        all_records: list[dict] = []
        progress = st.progress(0.0)
        for i, uploaded in enumerate(files):
            records = caption_pages(
                load_upload(uploaded.name, uploaded.getvalue()),
                model=model,
                visual_mode=visual_mode,
            )
            all_records.extend(records)
            progress.progress((i + 1) / len(files))
        save_history(all_records)
        records_table(all_records)
        st.download_button(
            "Download DocCaptionNet batch CSV",
            pd.DataFrame(all_records).to_csv(index=False).encode("utf-8"),
            file_name="DocCaptionNet_batch.csv",
            mime="text/csv",
        )

elif page == "History":
    st.markdown(
        """
        <div class="dcn-section">
          <h2>Caption history</h2>
          <p>Review captions generated in this workspace.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    records_table(load_history())

else:
    manifest = ROOT / "FILE_MANIFEST.md"
    st.markdown(
        """
        <div class="dcn-section">
          <h2>About DocCaptionNet</h2>
          <p>Project documentation and source files.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if manifest.exists():
        st.markdown(manifest.read_text(encoding="utf-8"))
