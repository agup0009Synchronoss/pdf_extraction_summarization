# gradio_app_demo.py
"""
Gradio UI for PDF Document Summarization & Labeling (Business Demo)

Features:
1. Upload up to N (default=5) PDF files at once.
2. For each PDF, run the pdf_pipeline to extract content and generate summaries.
3. Clean, business-friendly interface focusing on document summaries and classification.
4. Interactive UI to select processed documents and view:
   • LLM Summary & Classification
   • PDF Preview

Run with:
    python gradio_app_demo.py
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

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

# In-memory cache for the current Gradio session
CACHE: Dict[str, Dict[str, Any]] = {}
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

def _process_single_pdf(pdf_path: Path) -> Dict[str, Any]:
    """Run the pipeline on a single PDF (if not already cached) and return meta."""
    stem = pdf_path.stem
    if stem in CACHE:
        return CACHE[stem]

    # Ensure file is in a stable location (copy to output if uploaded tmp file)
    stable_path = OUTPUT_DIR / pdf_path.name
    if pdf_path != stable_path:
        shutil.copy2(pdf_path, stable_path)
        pdf_path = stable_path

    try:
        # Run the pipeline (no overwrite – we want caching on disk as well)
        pipeline_run(pdf_path, OUTPUT_DIR, overwrite=False)
    except Exception as e:
        # Handle pipeline failures gracefully
        meta = {
            "pdf_path": str(pdf_path),
            "extract_json": "",
            "prompt_json": "",
            "summary": f"Processing failed: {str(e)}",
            "label": "error",
            "prompt_content": "",
            "error": True
        }
        CACHE[stem] = meta
        return meta

    # Derive expected JSON paths
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
            "error": True
        }
        CACHE[stem] = meta
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
        "error": False
    }
    CACHE[stem] = meta
    return meta

# ---------------------------------------------------------------------------
# Gradio Callbacks

def process_pdfs(files: List[str], session_docs: List[str]) -> tuple[gr.Dropdown, str, List[str]]:
    if not files:
        return gr.update(), "No files selected.", session_docs
    if len(files) > MAX_FILES:
        return gr.update(), f"Please upload at most {MAX_FILES} PDFs at once.", session_docs

    # Always reset session list for demo (no caching)
    session_docs = []

    for file_path in files:
        meta = _process_single_pdf(Path(file_path))
        stem = Path(file_path).stem
        if stem not in session_docs:
            session_docs.append(stem)

    latest = session_docs[-1] if session_docs else None
    return gr.update(choices=session_docs, value=latest), f"Processed {len(files)} file(s).", session_docs


def show_document(doc_key: str, session_docs: List[str]):
    if not doc_key or doc_key not in session_docs:
        return None, "", ""
    data = CACHE[doc_key]
    
    pdf_html = create_pdf_viewer_html(data.get("pdf_path", ""))
    
    return pdf_html, data["summary"], data["label"]

# ---------------------------------------------------------------------------
# Build UI
with gr.Blocks(title="PDF Document Summarization & Labeling") as demo:
    gr.Markdown("## 📄 PDF Document Summarization & Labeling")
    gr.Markdown(
        "Upload PDF documents (max 5) to automatically generate summaries and classification labels. "
    )

    with gr.Row():
        file_uploader = gr.File(type="filepath", file_count="multiple", label="Upload PDF Documents")
        status = gr.Textbox(label="Status", interactive=False)

    # Per-session list of document stems
    docs_state = gr.State([])

    docs_dropdown = gr.Dropdown(label="Processed Documents", choices=[])

    pdf_viewer = gr.HTML(label="Document Preview")

    with gr.Row():
        summary_out  = gr.Textbox(label="Document Summary", lines=8, show_copy_button=True)
        label_out    = gr.Textbox(label="Document Classification", show_copy_button=True)

    # Wiring callbacks
    file_uploader.upload(process_pdfs,
                        inputs=[file_uploader, docs_state],
                        outputs=[docs_dropdown, status, docs_state])

    docs_dropdown.change(show_document,
                        inputs=[docs_dropdown, docs_state],
                        outputs=[pdf_viewer, summary_out, label_out])

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PDF Document Summarization & Labeling Demo")
    parser.add_argument("--share", action="store_true", help="Enable Gradio share link")
    parser.add_argument("--port", type=int, default=7862, help="Gradio server port")
    args = parser.parse_args()

    demo.launch(
        server_name="0.0.0.0",  # Allow external connections
        server_port=args.port,       # Use different port to avoid conflicts
        share=args.share,            # Disable share link due to network issues
        show_error=True         # Show detailed errors
    ) 