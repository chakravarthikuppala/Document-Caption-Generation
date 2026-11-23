# DocCaptionNet
"""Build a unique, content-based caption from extracted page text."""

from __future__ import annotations

import re

_HEADING_HINTS = (
    ("table of contents", "a table of contents"),
    ("contents", "a table of contents"),
    ("abstract", "an abstract page"),
    ("introduction", "an introduction page"),
    ("acknowledgement", "an acknowledgements page"),
    ("acknowledgment", "an acknowledgements page"),
    ("methodology", "a methodology page"),
    ("literature", "a literature-review page"),
    ("experiment", "an experiments page"),
    ("result", "a results page"),
    ("discussion", "a discussion page"),
    ("conclusion", "a conclusion page"),
    ("reference", "a references page"),
    ("bibliography", "a bibliography page"),
    ("appendix", "an appendix page"),
    ("invoice", "an invoice page"),
    ("receipt", "a receipt page"),
)


def _normalize(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def _title_line(text: str) -> str:
    for raw in text.splitlines():
        line = " ".join(raw.split())
        if 8 <= len(line) <= 120 and not line.endswith(","):
            return line
    words = text.split()
    return " ".join(words[:12]) if words else ""


def _kind(text: str) -> str:
    head = text[:900].lower()
    for needle, label in _HEADING_HINTS:
        if needle in head:
            return label
    return "a document page"


def caption_from_text(text: str, max_len: int = 320) -> str:
    cleaned = _normalize(text or "")
    if len(re.sub(r"\s+", "", cleaned)) < 12:
        return ""
    kind = _kind(cleaned)
    title = _title_line(cleaned)
    body = " ".join(cleaned.split())
    if title and body.lower().startswith(title.lower()):
        remainder = body[len(title) :].strip(" :-–")
    else:
        remainder = body
    snippet = remainder[:220].strip()
    if len(remainder) > 220:
        snippet = snippet.rsplit(" ", 1)[0] + "…"
    parts = [kind]
    if title:
        parts.append(f'titled "{title}"')
    sentence = " ".join(parts)
    if snippet:
        sentence = f"{sentence}: {snippet}"
    sentence = " ".join(sentence.split())
    if len(sentence) > max_len:
        sentence = sentence[: max_len - 1].rsplit(" ", 1)[0] + "…"
    return sentence


def compose_raw_caption(*, visual: str, page_text: str, backend: str) -> str:
    text_cap = caption_from_text(page_text)
    visual = " ".join((visual or "").split())
    generic_visual = "printed layout suitable for captioning" in visual
    if text_cap and backend == "blip" and visual and not generic_visual:
        return f"{text_cap} The page looks like {visual}."
    if text_cap:
        return text_cap
    if visual:
        return visual
    return "No caption was generated."
