# gradio_app.py
"""
Gradio UI for PDF Processing Pipeline

Features:
1. Upload up to N (default=5) PDF files at once.
2. For each PDF, run the pdf_pipeline (using SSL bypass patches via run_with_ssl_bypass).
3. Cache results during the session to avoid re-processing.
4. Interactive UI to select processed documents and visualise:
   • LLM Summary & Classification
   • Extraction JSON (pretty-printed)
   • Prompt + Response JSON (pretty-printed)

Run with:
    python gradio_app.py
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import List, Dict, Any
import tempfile
import base64
import argparse

import gradio as gr

# ---------------------------------------------------------------------------
# Apply SSL bypass patches for corporate environments
import os
import ssl
import urllib3
import warnings
import requests

# Disable SSL verification
os.environ['PYTHONHTTPSVERIFY'] = '0'
os.environ['REQUESTS_CA_BUNDLE'] = ''
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['SSL_CERT_FILE'] = ''
os.environ['HF_HUB_DISABLE_SSL_VERIFICATION'] = '1'
os.environ['HF_HOME'] = './hf_cache'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings('ignore', message='Unverified HTTPS request')

# Create unverified SSL context
ssl._create_default_https_context = ssl._create_unverified_context

# Patch requests to use unverified SSL
class UnverifiedSession(requests.Session):
    def __init__(self):
        super().__init__()
        self.verify = False

# Monkey patch requests.Session
requests.Session = UnverifiedSession

from pdf_pipeline import run as pipeline_run
from config import DEFAULT_CONFIG

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

# In-memory cache for the current Gradio session
CACHE: Dict[str, Dict[str, Any]] = {}
DOC_SELECTION_MAP: Dict[str, str] = {}  # dropdown label -> cache key
MAX_FILES = 5  # Maximum PDFs that can be uploaded at once

# ---------------------------------------------------------------------------
# Helper functions

def create_pdf_viewer_html(pdf_path: str) -> str:
    """Return HTML string that embeds the PDF via data URI so browser can render it."""
    if not pdf_path or not Path(pdf_path).exists():
        return "<p>PDF not available</p>"

    try:
        # Read bytes and encode as base64
        pdf_bytes = Path(pdf_path).read_bytes()
        b64_pdf   = base64.b64encode(pdf_bytes).decode('ascii')
        data_uri  = f"data:application/pdf;base64,{b64_pdf}"

        html_content = f"""
        <div style='width: 100%; height: 600px; border: 1px solid #ccc;'>
            <embed src='{data_uri}' type='application/pdf' width='100%' height='100%'/>
        </div>
        """
        return html_content
    except Exception as exc:
        # Fallback: provide download link
        return f"<p>Unable to display PDF. <a href='{pdf_path}' download>Download</a> (error: {exc})</p>"

def _process_single_pdf(
    pdf_path: Path,
    enable_ocr: bool = True,
    ocr_engine: str = "paddleocr",
    ocr_language: str = "en",
    min_words_trigger: int = 200,
    min_image_area_pixels: int = 50000,
    min_image_area_percent: float = 0.15,
    max_images: int = 5,
) -> Dict[str, Any]:
    """
    Run the pipeline on a single PDF (if not already cached) and return meta.
    
    Args:
        pdf_path: Path to PDF file
        enable_ocr: Whether to enable OCR on images
        ocr_language: OCR language code
        
    Returns:
        Metadata dict with processing results
    """
    # Create cache key including OCR settings
    cache_key = (
        f"{pdf_path.stem}:ocr={enable_ocr}:engine={ocr_engine}:lang={ocr_language}:"
        f"minw={min_words_trigger}:minpx={min_image_area_pixels}:minpct={min_image_area_percent}:maximg={max_images}"
    )
    
    if cache_key in CACHE:
        return CACHE[cache_key]

    # Ensure file is in a stable location (copy to output if uploaded tmp file)
    stable_path = OUTPUT_DIR / pdf_path.name
    if pdf_path != stable_path:
        shutil.copy2(pdf_path, stable_path)
        pdf_path = stable_path

    # Build config overrides
    config_overrides = {
        "MIN_WORDS": int(min_words_trigger),
        "MIN_IMAGE_AREA_PIXELS": int(min_image_area_pixels),
        "MIN_IMAGE_AREA_PERCENT": float(min_image_area_percent),
        "MAX_IMAGES": int(max_images),
        "ENABLE_IMAGE_OCR": bool(enable_ocr),
        "OCR_ENGINE": ocr_engine,
        "OCR_LANGUAGES": [ocr_language],
    }

    try:
        # Run the pipeline with config overrides
        pipeline_run(pdf_path, OUTPUT_DIR, overwrite=False, config_overrides=config_overrides)
    except Exception as e:
        # Handle pipeline failures gracefully
        meta = {
            "pdf_path": str(pdf_path),
            "extract_json": "",
            "prompt_json": "",
            "summary": f"Processing failed: {str(e)}",
            "label": "error",
            "prompt_content": "",
            "error": True,
            "cache_key": cache_key
        }
        CACHE[cache_key] = meta
        return meta

    # Derive expected JSON paths
    stem = pdf_path.stem
    extract_json = OUTPUT_DIR / f"{stem}_dual.json"
    prompt_json = OUTPUT_DIR / f"{stem}_prompt_response.json"

    # Check if files exist before trying to read them
    if not prompt_json.exists():
        meta = {
            "pdf_path": str(pdf_path),
            "extract_json": str(extract_json) if extract_json.exists() else "",
            "prompt_json": "",
            "summary": "Processing completed but response file not found",
            "label": "unknown",
            "prompt_content": "",
            "error": True,
            "cache_key": cache_key
        }
        CACHE[cache_key] = meta
        return meta

    try:
        # Load prompt/response JSON to extract summary & label
        with open(prompt_json, "r", encoding="utf-8") as fp:
            prompt_data = json.load(fp)

        summary = prompt_data.get("response_body", {}).get("content", {}).get("summary", "?")
        label = prompt_data.get("response_body", {}).get("content", {}).get("classification_label", "?")
    except Exception as e:
        summary = f"Error reading response: {str(e)}"
        label = "error"

    meta = {
        "pdf_path": str(pdf_path),
        "extract_json": str(extract_json),
        "prompt_json": str(prompt_json),
        "summary": summary,
        "label": label,
        "prompt_content": prompt_data.get("llm_prompt", {}).get("content", "") if 'prompt_data' in locals() else "",
        "error": False,
        "cache_key": cache_key
    }
    CACHE[cache_key] = meta
    return meta

# ---------------------------------------------------------------------------
# Gradio Callbacks

def process_pdfs(
    files: List[str],
    caching_enabled: bool,
    enable_ocr: bool,
    ocr_engine: str,
    ocr_language: str,
    min_words_trigger: int,
    min_image_area_pixels: int,
    min_image_area_percent: float,
    max_images: int,
    session_docs: List[str]
) -> tuple[gr.Dropdown, str, List[str]]:
    """
    Process uploaded PDFs with current OCR settings.
    
    Args:
        files: List of file paths
        caching_enabled: Whether to keep session cache
        enable_ocr: Whether to enable OCR
        ocr_language: OCR language code
        session_docs: Current session documents
        
    Returns:
        Updated dropdown, status message, updated session docs
    """
    if not files:
        return gr.update(), "No files selected.", session_docs
    if len(files) > MAX_FILES:
        return gr.update(), f"Please upload at most {MAX_FILES} PDFs at once.", session_docs

    # Reset session list if caching not enabled
    if not caching_enabled:
        session_docs = []

    for file_path in files:
        meta = _process_single_pdf(
            Path(file_path),
            enable_ocr=enable_ocr,
            ocr_engine=ocr_engine,
            ocr_language=ocr_language,
            min_words_trigger=min_words_trigger,
            min_image_area_pixels=min_image_area_pixels,
            min_image_area_percent=min_image_area_percent,
            max_images=max_images,
        )
        stem = Path(file_path).stem
        display_name = (
            f"{stem} [ocr={'on' if enable_ocr else 'off'}, "
            f"engine={ocr_engine}, lang={ocr_language}]"
        )
        DOC_SELECTION_MAP[display_name] = meta["cache_key"]
        if display_name not in session_docs:
            session_docs.append(display_name)

    latest = session_docs[-1] if session_docs else None
    ocr_status = (
        f" (OCR: {'ON' if enable_ocr else 'OFF'}, Engine: {ocr_engine}, Lang: {ocr_language})"
        if enable_ocr else " (OCR: OFF)"
    )
    return gr.update(choices=session_docs, value=latest), f"Processed {len(files)} file(s).{ocr_status}", session_docs


def show_document(doc_key: str, session_docs: List[str]):
    if not doc_key or doc_key not in session_docs:
        return None, "", "", "", "", ""

    cache_key = DOC_SELECTION_MAP.get(doc_key)
    if not cache_key or cache_key not in CACHE:
        return None, "", "", "", "", "Selected document not found in cache"

    data = CACHE[cache_key]
    
    pdf_html = create_pdf_viewer_html(data.get("pdf_path", ""))

    # Handle extraction JSON
    extract_str = ""
    if data.get("extract_json") and Path(data["extract_json"]).exists():
        try:
            with open(data["extract_json"], "r", encoding="utf-8") as f1:
                extract_str = json.dumps(json.load(f1), indent=2, ensure_ascii=False)
        except Exception as e:
            extract_str = f"Error reading extraction JSON: {str(e)}"
    else:
        extract_str = "Extraction JSON not available"
    
    # Handle prompt JSON and plain text
    prompt_str = ""
    prompt_text = data.get("prompt_content", "")
    if data.get("prompt_json") and Path(data["prompt_json"]).exists():
        try:
            with open(data["prompt_json"], "r", encoding="utf-8") as f2:
                prompt_str = json.dumps(json.load(f2), indent=2, ensure_ascii=False)
        except Exception as e:
            prompt_str = f"Error reading prompt JSON: {str(e)}"
    else:
        prompt_str = "Prompt JSON not available"
    
    return pdf_html, data["summary"], data["label"], prompt_str, prompt_text, extract_str

# ---------------------------------------------------------------------------
# Build UI
with gr.Blocks(title="PDF Pipeline UI") as demo:
    gr.Markdown("## 📄 PDF Processing Pipeline – Gradio UI")
    gr.Markdown(
        "Upload PDF(s) (max 5) to run extraction + LLM summarisation. "
        "Processed documents are cached for this session so you can "
        "quickly revisit results without re-processing."
    )

    with gr.Row():
        file_uploader = gr.File(type="filepath", file_count="multiple", label="Upload PDF(s)")
        status = gr.Textbox(label="Status", interactive=False)
        caching_checkbox = gr.Checkbox(label="📌 Enable caching", value=False)
    
    # OCR Extraction Options
    with gr.Accordion("⚙️ Extraction Options", open=False):
        with gr.Row():
            enable_ocr = gr.Checkbox(
                label="Enable Image OCR",
                value=DEFAULT_CONFIG.ENABLE_IMAGE_OCR,
                info="Extract text from PDF-embedded images"
            )
            ocr_engine = gr.Dropdown(
                choices=["paddleocr", "easyocr"],
                value=DEFAULT_CONFIG.OCR_ENGINE,
                label="OCR Engine",
                info="Default is PaddleOCR; EasyOCR kept as fallback"
            )
            ocr_language = gr.Dropdown(
                choices=["en", "ch", "fr", "de", "es", "it", "ja", "ko", "ru"],
                value="en",
                label="OCR Language",
                info="Primary language code"
            )
        with gr.Row():
            min_words_trigger = gr.Number(
                value=DEFAULT_CONFIG.MIN_WORDS,
                label="Image Extraction Trigger (min words)",
                precision=0,
                minimum=0,
                info="If docling word count is below this, image extraction is enabled"
            )
            min_image_area_pixels = gr.Number(
                value=DEFAULT_CONFIG.MIN_IMAGE_AREA_PIXELS,
                label="Min Image Area (pixels)",
                precision=0,
                minimum=1,
                info="Absolute size floor"
            )
            min_image_area_percent = gr.Slider(
                minimum=0.01,
                maximum=1.0,
                value=DEFAULT_CONFIG.MIN_IMAGE_AREA_PERCENT,
                step=0.01,
                label="Min Relative Image/Page Area",
                info="Relative size filter against estimated page area"
            )
            max_images = gr.Number(
                value=DEFAULT_CONFIG.MAX_IMAGES,
                label="Max Images",
                precision=0,
                minimum=1,
                info="Upper bound of image elements passed to LLM"
            )

    # Per-session list of document stems
    docs_state = gr.State([])

    docs_dropdown = gr.Dropdown(label="Processed Documents", choices=[])

    pdf_viewer = gr.HTML(label="PDF Preview")

    with gr.Row():
        summary_out  = gr.Textbox(label="LLM Summary")
        label_out    = gr.Textbox(label="Classification Label")

    with gr.Tabs():
        with gr.TabItem("Prompt & Response (JSON)"):
            prompt_json_tab  = gr.Code(label="LLM Prompt & Response JSON", language="json")
        with gr.TabItem("Prompt & Response (Text)"):
            prompt_text_tab = gr.Textbox(label="LLM Prompt (plain text)", lines=10)
        with gr.TabItem("Extraction JSON"):
            extract_json = gr.Code(label="Document Extraction Data", language="json")

    # Wiring callbacks
    file_uploader.upload(
        process_pdfs,
        inputs=[
            file_uploader,
            caching_checkbox,
            enable_ocr,
            ocr_engine,
            ocr_language,
            min_words_trigger,
            min_image_area_pixels,
            min_image_area_percent,
            max_images,
            docs_state,
        ],
        outputs=[docs_dropdown, status, docs_state]
    )

    docs_dropdown.change(show_document,
                        inputs=[docs_dropdown, docs_state],
                        outputs=[pdf_viewer, summary_out, label_out, prompt_json_tab, prompt_text_tab, extract_json])

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gradio UI for PDF Processing Pipeline")
    parser.add_argument("--share", action="store_true", help="Enable Gradio share link")
    parser.add_argument("--port", type=int, default=7861, help="Gradio server port")
    args = parser.parse_args()

    demo.launch(
        server_name="0.0.0.0",  # Allow external connections
        server_port=args.port,       # Use different port to avoid conflicts
        share=args.share,            # Disable share link due to network issues
        show_error=True         # Show detailed errors
    ) 