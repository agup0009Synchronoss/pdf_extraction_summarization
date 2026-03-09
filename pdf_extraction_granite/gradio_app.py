"""
Gradio UI for Granite-Docling PDF Extraction POC
=================================================

Pipeline:
  1. Upload a PDF
  2. Docling VLM pipeline (granite_docling preset) → Markdown + HTML + JSON
  3. Build a structured prompt from DoclingDocument elements (titles, sections,
     paragraphs, tables, captions — each tagged by type and page number)
  4. POST prompt to remote Ollama service
  5. Display summary, classification, Markdown preview, HTML preview,
     Extraction JSON, Prompt preview, ZIP download

Run with:
    python gradio_app.py
    python gradio_app.py --port 7863 --share
"""

# ---------------------------------------------------------------------------
# SSL bypass — must run before any HuggingFace / requests / httpx imports.
import os
import ssl
import urllib3
import warnings
import requests

os.environ['PYTHONHTTPSVERIFY'] = '0'
os.environ['REQUESTS_CA_BUNDLE'] = ''
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['SSL_CERT_FILE'] = ''
os.environ['HF_HUB_DISABLE_SSL_VERIFICATION'] = '1'
os.environ['HF_HOME'] = './hf_cache'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
# Disable HuggingFace XET (Rust-level large-file downloader) — it ignores
# Python SSL patches and panics on corporate networks with SSL inspection.
# Falling back to regular Python HTTPS which our patches above already cover.
os.environ['HF_HUB_DISABLE_XET'] = '1'

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings('ignore', message='Unverified HTTPS request')

ssl._create_default_https_context = ssl._create_unverified_context


class _UnverifiedSession(requests.Session):
    def __init__(self):
        super().__init__()
        self.verify = False


requests.Session = _UnverifiedSession

try:
    import httpx

    _orig_client = httpx.Client.__init__

    def _patched_client(self, *args, **kwargs):
        kwargs['verify'] = False
        _orig_client(self, *args, **kwargs)

    httpx.Client.__init__ = _patched_client

    _orig_async_client = httpx.AsyncClient.__init__

    def _patched_async_client(self, *args, **kwargs):
        kwargs['verify'] = False
        _orig_async_client(self, *args, **kwargs)

    httpx.AsyncClient.__init__ = _patched_async_client
except ImportError:
    pass
# ---------------------------------------------------------------------------

import argparse
import logging
import tempfile
import zipfile
from pathlib import Path

import gradio as gr

from call_llama_api import API_URL, MODEL
from granite_pipeline import (
    CLASSIFICATION_LABELS,
    build_prompt,
    call_ollama,
    run_docling,
)

# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("granite_gradio")

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Main Gradio callback

def convert_pdf(pdf_file):
    """
    Process an uploaded PDF through the full pipeline.

    Returns:
        status, summary, classification, md_text, html_text,
        json_text, zip_path, prompt_text, logs_str
    """
    _empty = ("", "", "", "", "", "", None, "", "")

    if not pdf_file:
        return ("No file uploaded.",) + _empty[1:]

    pdf_path = str(pdf_file)
    stem = Path(pdf_path).stem
    filename = Path(pdf_path).name

    run_dir = Path(tempfile.mkdtemp(prefix=f"granite_{stem}_"))
    out_dir = run_dir / "docling_out"

    logs = []

    # --- Stage 1: Docling extraction ---
    logs.append(f"[Stage 1] Docling VLM pipeline on: {filename}")
    try:
        doc = run_docling(pdf_path, out_dir)
        logs.append("  Docling OK.")
    except Exception as exc:
        logs.append(f"  [FAIL] {exc}")
        logs_str = "\n".join(logs)
        return logs_str, "", "", "", "", "", None, "", logs_str

    # Read exported files
    md_files = sorted(out_dir.glob("*.md"))
    html_files = sorted(out_dir.glob("*.html"))
    json_files = sorted(out_dir.glob("*.json"))

    md_text = md_files[0].read_text(encoding="utf-8", errors="replace") if md_files else ""
    html_text = html_files[0].read_text(encoding="utf-8", errors="replace") if html_files else ""
    json_text = json_files[0].read_text(encoding="utf-8", errors="replace") if json_files else ""

    logs.append(f"  Markdown: {md_files[0].name if md_files else 'none'}")
    logs.append(f"  HTML:     {html_files[0].name if html_files else 'none'}")
    logs.append(f"  JSON:     {json_files[0].name if json_files else 'none'}")

    # --- Stage 2: Build structured prompt ---
    logs.append("[Stage 2] Building structured prompt from DoclingDocument elements ...")
    prompt_text = build_prompt(doc, filename)
    logs.append(f"  Prompt length: {len(prompt_text.split())} words")

    # --- Stage 3: Call Ollama ---
    logs.append(f"[Stage 3] Sending prompt to Ollama ({MODEL}) ...")
    try:
        llm_result = call_ollama(prompt_text)
        summary = llm_result["summary"]
        classification = llm_result["classification_label"]
        logs.append(f"  Classification: {classification}")
        logs.append("  Ollama OK.")
    except Exception as exc:
        logs.append(f"  [FAIL] Ollama error: {exc}")
        summary = f"LLM call failed: {exc}"
        classification = "error"

    # --- Stage 4: ZIP all outputs ---
    zip_path = run_dir / f"{stem}_docling_outputs.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in out_dir.rglob("*"):
            if p.is_file():
                zf.write(p, arcname=p.relative_to(out_dir))
    logs.append(f"[Stage 4] ZIP: {zip_path.name}")

    status = (
        f"Done: {filename}\n"
        f"  Classification: {classification}\n"
        f"  Pages: {len(doc.pages) if doc.pages else '?'}\n"
        f"  LLM: {MODEL} @ {API_URL}"
    )

    logs_str = "\n".join(logs)
    return status, summary, classification, md_text, html_text, json_text, str(zip_path), prompt_text, logs_str


# ---------------------------------------------------------------------------
# Gradio UI

with gr.Blocks(title="Granite-Docling PDF Extraction POC") as demo:
    gr.Markdown("# Granite-Docling-258M PDF Extraction POC")
    gr.Markdown(
        "Upload a PDF. The Docling VLM pipeline (`granite_docling` preset) extracts the "
        "document into structured elements — titles, sections, paragraphs, tables, captions — "
        "each tagged by type and page number. The structured content is sent to the remote "
        f"Ollama service (`{MODEL}`) for summarisation and classification."
    )

    with gr.Row():
        file_uploader = gr.File(
            label="Upload PDF",
            file_count="single",
            type="filepath",
        )

    process_button = gr.Button("Process PDF", variant="primary", size="lg")

    status_output = gr.Textbox(label="Status", interactive=False, lines=4)

    gr.Textbox(
        label="LLM Endpoint",
        value=f"{MODEL} @ {API_URL}",
        interactive=False,
        lines=1,
    )

    gr.Markdown("## Results")

    with gr.Row():
        summary_output = gr.Textbox(label="Summary", lines=4, interactive=False)
        classification_output = gr.Textbox(label="Classification", interactive=False)

    with gr.Tabs():
        with gr.TabItem("Markdown Preview"):
            md_output = gr.Markdown(label="Extracted Markdown")

        with gr.TabItem("HTML Split-Page Preview"):
            html_output = gr.HTML(label="Docling split-page HTML")

        with gr.TabItem("Extraction JSON"):
            json_output = gr.Code(label="DoclingDocument JSON", language="json", lines=30)

        with gr.TabItem("Prompt sent to LLM"):
            prompt_output = gr.Textbox(label="Full prompt", lines=20, interactive=False)

    zip_output = gr.File(label="Download all Docling outputs (ZIP)")

    logs_output = gr.Textbox(label="Processing Logs", lines=10, interactive=False)

    process_button.click(
        fn=convert_pdf,
        inputs=[file_uploader],
        outputs=[
            status_output,
            summary_output,
            classification_output,
            md_output,
            html_output,
            json_output,
            zip_output,
            prompt_output,
            logs_output,
        ],
    )


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Granite-Docling PDF Extraction POC")
    parser.add_argument("--share", action="store_true", help="Enable Gradio public share link")
    parser.add_argument("--port", type=int, default=7863, help="Gradio server port")
    args = parser.parse_args()

    log.info("Starting Granite-Docling Gradio application")
    log.info("  LLM endpoint: %s", API_URL)
    log.info("  LLM model:    %s", MODEL)

    demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
        show_error=True,
    )
