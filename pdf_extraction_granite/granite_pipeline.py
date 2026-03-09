"""
granite_pipeline.py
===================
Pipeline logic for the Granite-Docling PDF extraction POC.

Three public functions:
  run_docling(pdf_path, out_dir)  ->  DoclingDocument
  build_prompt(doc, filename)     ->  str
  call_ollama(prompt)             ->  dict {summary, classification_label}
"""

import json
import logging
import textwrap
from pathlib import Path
from typing import Any

from docling_core.types.doc.document import (
    DocItemLabel,
    PictureItem,
    TableItem,
)

from call_llama_api import API_KEY, API_URL, MODEL, send_prompt

log = logging.getLogger("granite_pipeline")

# ---------------------------------------------------------------------------
# Constants

MAX_PROMPT_WORDS = 2000

CLASSIFICATION_LABELS = [
    "scanned_document",
    "handwritten_form",
    "receipt",
    "invoice",
    "tax_form",
    "contract",
    "medical_record",
    "id_document",
    "financial_statement",
    "correspondence",
    "unclassified",
]

# Labels we skip in the prompt (furniture / noise)
_SKIP_LABELS = {
    DocItemLabel.PAGE_HEADER,
    DocItemLabel.PAGE_FOOTER,
    DocItemLabel.EMPTY_VALUE,
}


# ---------------------------------------------------------------------------
# Step 1: Run Docling VLM pipeline

def run_docling(pdf_path: str, out_dir: Path) -> Any:
    """
    Convert a PDF with the Granite-Docling VLM pipeline (in-process).

    Exports to out_dir:
      - {stem}.md   -- clean Markdown (image placeholders, no base64)
      - {stem}.html -- split-page HTML
      - {stem}.json -- full DoclingDocument JSON

    Returns the DoclingDocument object.
    Raises RuntimeError on failure.
    """
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import VlmConvertOptions, VlmPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.pipeline.vlm_pipeline import VlmPipeline

    log.info("Starting Docling VLM conversion: %s", pdf_path)

    try:
        vlm_options = VlmConvertOptions.from_preset("granite_docling")
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_cls=VlmPipeline,
                    pipeline_options=VlmPipelineOptions(vlm_options=vlm_options),
                ),
            }
        )
        result = converter.convert(source=pdf_path)
        doc = result.document

        stem = Path(pdf_path).stem
        out_dir.mkdir(parents=True, exist_ok=True)

        # Markdown: default image_mode is PLACEHOLDER (no base64 blobs)
        md_path = out_dir / f"{stem}.md"
        md_path.write_text(doc.export_to_markdown(), encoding="utf-8")

        # HTML
        html_path = out_dir / f"{stem}.html"
        html_path.write_text(doc.export_to_html(), encoding="utf-8")

        # Structured JSON
        json_path = out_dir / f"{stem}.json"
        json_path.write_text(
            json.dumps(doc.export_to_dict(), indent=2, default=str),
            encoding="utf-8",
        )

        log.info("Docling OK -> %s.{md,html,json}", stem)
        return doc

    except Exception as exc:
        raise RuntimeError(f"Docling conversion failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Step 2: Build structured prompt from DoclingDocument

def build_prompt(doc: Any, filename: str) -> str:
    """
    Build an LLM prompt by walking the DoclingDocument in reading order.

    Each element is tagged by semantic type:
        [Page N - TITLE]
        text...

        [Page N - SECTION_HEADER]
        text...

        [Page N - TABLE]
        | col | col |
        ...

    This preserves document structure for the LLM, unlike dumping raw markdown.
    Truncates total content to MAX_PROMPT_WORDS.
    """
    page_count = len(doc.pages) if doc.pages else "?"

    labels_str = ", ".join(CLASSIFICATION_LABELS)

    header = textwrap.dedent(f"""\
        You are an expert document analyst specialising in digitised documents.

        TASKS:
        1. Summarise the document in 2-3 crisp sentences that capture the key information.
        2. Classify the document into exactly ONE category from the list below.

        ALLOWED CATEGORIES:
        {labels_str}

        Return ONLY valid JSON — no markdown, no extra commentary:
        {{
          "summary": "2-3 sentence summary here",
          "classification_label": "one_category_from_list_above"
        }}

        DOCUMENT CONTENT:

        [Document: {filename}]
        [Total Pages: {page_count}]
        [Extraction Method: Docling VLM / granite_docling]

        """)

    content_parts: list[str] = []
    total_words = len(header.split())
    budget = MAX_PROMPT_WORDS - total_words - 30  # small reserve

    for item, _level in doc.iterate_items():
        if item.label in _SKIP_LABELS:
            continue

        # Determine page number
        page_no = item.prov[0].page_no if item.prov else "?"

        # Format content based on element type
        if isinstance(item, TableItem):
            try:
                cell_text = item.export_to_markdown(doc)
            except Exception:
                cell_text = "[table]"
            chunk = f"\n[Page {page_no} - TABLE]\n{cell_text}\n"

        elif isinstance(item, PictureItem):
            # Use caption text if available, skip bare images
            captions = [c.resolve(doc).text for c in item.captions if c.resolve(doc).text.strip()]
            if captions:
                chunk = f"\n[Page {page_no} - PICTURE CAPTION]\n{' '.join(captions)}\n"
            else:
                continue  # no caption, nothing useful to send

        else:
            text = getattr(item, "text", "").strip()
            if not text:
                continue
            label_name = item.label.value.upper().replace("_", " ")
            chunk = f"\n[Page {page_no} - {label_name}]\n{text}\n"

        chunk_words = len(chunk.split())
        if total_words + chunk_words > MAX_PROMPT_WORDS:
            remaining = budget - (total_words - len(header.split()))
            if remaining > 0:
                words = chunk.split()[:remaining]
                content_parts.append(" ".join(words) + "\n\n[Content truncated]")
            break

        content_parts.append(chunk)
        total_words += chunk_words

    if not content_parts:
        content_parts.append("[No extractable content found]")

    log.info("Prompt built: %d words, %d elements", total_words, len(content_parts))
    return header + "".join(content_parts)


# ---------------------------------------------------------------------------
# Step 3: Call the Ollama LLM

def call_ollama(prompt: str) -> dict:
    """
    Send the prompt to the shared Ollama endpoint.

    Returns dict with 'summary' and 'classification_label'.
    Raises Exception on API or parse failure.
    """
    log.info("Calling LLM: %s, model=%s, ~%d words", API_URL, MODEL, len(prompt.split()))
    response = send_prompt(prompt, api_key=API_KEY)

    if response.status_code != 200:
        raise Exception(
            f"LLM API returned HTTP {response.status_code}: {response.text[:400]}"
        )

    try:
        body = response.json()
    except Exception as exc:
        raise Exception(f"Could not decode LLM response as JSON: {exc}") from exc

    content = body.get("message", {}).get("content", "")
    if not content:
        raise Exception("LLM response has empty 'message.content'")

    parsed = None
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        if "```json" in content:
            start = content.find("```json") + 7
            end = content.find("```", start)
            if end > start:
                parsed = json.loads(content[start:end].strip())
        if parsed is None:
            raise Exception(f"LLM content is not valid JSON: {content[:300]}")

    summary = parsed.get("summary", "")
    classification = parsed.get("classification_label", "")

    if not summary or not classification:
        raise Exception(f"LLM response missing required fields. Got: {list(parsed.keys())}")

    if classification not in CLASSIFICATION_LABELS:
        log.warning("LLM returned label '%s' not in expected list.", classification)

    return {"summary": summary, "classification_label": classification}
