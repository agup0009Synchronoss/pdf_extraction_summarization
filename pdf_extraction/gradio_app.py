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

def process_pdfs(files: List[str], caching_enabled: bool, session_docs: List[str]) -> tuple[gr.Dropdown, str, List[str]]:
    if not files:
        return gr.update(), "No files selected.", session_docs
    if len(files) > MAX_FILES:
        return gr.update(), f"Please upload at most {MAX_FILES} PDFs at once.", session_docs

    # Reset session list if caching not enabled
    if not caching_enabled:
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
        return None, "", "", "", "", ""
    data = CACHE[doc_key]
    
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

    # Per-session list of document stems
    docs_state = gr.State([])

    docs_dropdown = gr.Dropdown(label="Processed Documents", choices=[])

    pdf_viewer = gr.HTML(label="PDF Preview")

    with gr.Row():
        summary_out  = gr.Textbox(label="LLM Summary", show_copy_button=True)
        label_out    = gr.Textbox(label="Classification Label")

    with gr.Tabs():
        with gr.TabItem("Prompt & Response (JSON)"):
            prompt_json_tab  = gr.Code(label="LLM Prompt & Response JSON", language="json")
        with gr.TabItem("Prompt & Response (Text)"):
            prompt_text_tab = gr.Textbox(label="LLM Prompt (plain text)", lines=10, show_copy_button=True)
        with gr.TabItem("Extraction JSON"):
            extract_json = gr.Code(label="Document Extraction Data", language="json")

    # Wiring callbacks
    file_uploader.upload(process_pdfs,
                        inputs=[file_uploader, caching_checkbox, docs_state],
                        outputs=[docs_dropdown, status, docs_state])

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