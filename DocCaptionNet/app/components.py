# DocCaptionNet
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT = "DocCaptionNet"
ASSETS = Path(__file__).resolve().parent / "assets"
LOGO = ASSETS / "DocCaptionNet_logo.svg"

CSS = """
<style>
  :root {
    --dcn-ink: #172b43;
    --dcn-muted: #607286;
    --dcn-brand: #164b69;
    --dcn-line: #dce5ed;
  }
  .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    background: #f3f6fa !important;
    color: var(--dcn-ink) !important;
  }
  [data-testid="stMainBlockContainer"] {
    max-width: 1160px;
    padding-top: 2rem;
    padding-bottom: 4rem;
  }
  .stApp p, .stApp span, .stApp label, .stApp h1, .stApp h2, .stApp h3,
  .stMarkdown, [data-testid="stMarkdownContainer"],
  [data-testid="stMarkdownContainer"] p { color: var(--dcn-ink) !important; }
  [data-testid="stHeader"] { background: transparent !important; }
  [data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid var(--dcn-line);
  }
  [data-testid="stSidebar"] p, [data-testid="stSidebar"] span,
  [data-testid="stSidebar"] label, [data-testid="stSidebar"] h1,
  [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3,
  [data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
  [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: var(--dcn-ink) !important;
  }
  [data-testid="stSidebar"] h3 {
    margin-bottom: 1.25rem;
    font-size: 1.05rem;
  }
  [data-testid="stSidebar"] [data-testid="stRadio"] > label {
    color: var(--dcn-muted) !important;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.045em;
    text-transform: uppercase;
  }
  [data-testid="stFileUploader"] section {
    background: #ffffff !important;
    border: 1px dashed #a9bac9 !important;
    border-radius: 14px !important;
    padding: 1.2rem !important;
  }
  [data-testid="stFileUploader"] button {
    background: var(--dcn-brand) !important;
    color: #ffffff !important;
    border: 1px solid var(--dcn-brand) !important;
    border-radius: 9px !important;
  }
  [data-testid="stFileUploader"] button span,
  [data-testid="stFileUploader"] button p {
    color: #ffffff !important;
  }
  [data-testid="stFileUploader"] small {
    color: var(--dcn-muted) !important;
  }
  .stButton button[kind="primary"] {
    background: var(--dcn-brand) !important;
    border: 1px solid var(--dcn-brand) !important;
    border-radius: 9px !important;
    font-weight: 600 !important;
    padding: 0.55rem 1.1rem !important;
  }
  .stButton button[kind="primary"] p,
  .stButton button[kind="primary"] span { color: #ffffff !important; }
  .dcn-brand {
    display: flex;
    align-items: center;
    min-height: 112px;
    padding: 1.4rem 1.7rem;
    background: #ffffff;
    border: 1px solid var(--dcn-line);
    border-left: 4px solid var(--dcn-brand);
    border-radius: 16px;
    box-shadow: 0 8px 26px rgba(23, 43, 67, 0.045);
  }
  .dcn-kicker {
    margin-bottom: 0.45rem;
    color: #55748d !important;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
  }
  .dcn-brand h1 {
    margin: 0;
    color: var(--dcn-ink) !important;
    font-size: 1.9rem;
    line-height: 1.2;
    letter-spacing: -0.035em;
  }
  .dcn-brand p {
    margin: 0.45rem 0 0;
    color: var(--dcn-muted) !important;
    font-size: 0.98rem;
    line-height: 1.5;
  }
  .dcn-section {
    margin: 2rem 0 0.85rem;
  }
  .dcn-section h2 {
    margin: 0;
    color: var(--dcn-ink) !important;
    font-size: 1.3rem;
    letter-spacing: -0.02em;
  }
  .dcn-section p {
    margin: 0.35rem 0 0;
    color: var(--dcn-muted) !important;
    font-size: 0.93rem;
  }
  .dcn-hero {
    background: #ffffff;
    color: var(--dcn-ink) !important;
    padding: 1.4rem 1.6rem;
    border-radius: 16px;
    border: 1px solid var(--dcn-line);
    margin-bottom: 1.2rem;
  }
  .dcn-card {
    background: #ffffff;
    border: 1px solid var(--dcn-line);
    border-radius: 14px;
    padding: 1rem 1.15rem;
    box-shadow: 0 6px 20px rgba(23, 43, 67, 0.05);
    margin-bottom: 0.85rem;
  }
  .dcn-caption {
    font-size: 1.05rem;
    line-height: 1.55;
    color: var(--dcn-ink) !important;
    font-weight: 600;
  }
  .dcn-meta {
    margin-top: 0.45rem;
    font-size: 0.85rem;
    color: var(--dcn-muted) !important;
  }
  @media (max-width: 700px) {
    [data-testid="stMainBlockContainer"] { padding: 1.2rem 1rem 3rem; }
    .dcn-brand { min-height: 0; padding: 1.2rem; }
    .dcn-brand h1 { font-size: 1.55rem; }
  }
</style>
"""


def inject_style() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def hero() -> None:
    logo_column, title_column = st.columns([0.12, 0.88], vertical_alignment="center")
    with logo_column:
        if LOGO.exists():
            st.image(str(LOGO), width=64)
    with title_column:
        st.markdown(
            f"""
            <div class="dcn-brand">
              <div>
                <div class="dcn-kicker">Document intelligence workspace</div>
                <h1>{PROJECT}</h1>
                <p>Generate clear, page-aware captions for documents, scans, and PDFs.</p>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def records_table(records: list[dict]) -> None:
    if not records:
        st.info("No DocCaptionNet captions yet.")
        return
    df = pd.DataFrame(records)
    show = [c for c in ["document_name", "page", "total_pages", "caption", "backend", "created_at"] if c in df.columns]
    st.dataframe(df[show], use_container_width=True, hide_index=True)
