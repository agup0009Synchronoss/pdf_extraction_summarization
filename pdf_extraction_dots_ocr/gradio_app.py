"""
Gradio UI for DOTS OCR Application
==================================

Moderate UI with key controls:
- PDF upload
- Page cap control
- DPI control
- Performance preset selector
- Device policy display
- Status updates
- Output panels for all artifacts
"""

import gradio as gr
from pathlib import Path
import json
import logging
import shutil
import tempfile
from typing import List, Dict, Any

from config import DotsConfig, get_config, DEFAULT_CONFIG
from pipeline import run_pipeline
from call_llama_api import API_URL, MODEL  # for display only

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger("gradio_app")

# Session cache
CACHE: Dict[str, Dict[str, Any]] = {}
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)


def process_pdf_upload(
    pdf_file,
    page_cap: int,
    dpi: int,
    performance_preset: str,
    device_policy: str
) -> tuple:
    """
    Process uploaded PDF through DOTS pipeline.
    
    Returns:
        Tuple of (status_text, summary, classification, raw_json, normalized_json, prompt_text, device_info)
    """
    if not pdf_file:
        return ("No file uploaded", "", "", "", "", "", "")
    
    try:
        # Copy to stable location
        pdf_path = Path(pdf_file)
        stable_path = OUTPUT_DIR / pdf_path.name
        shutil.copy2(pdf_path, stable_path)
        
        log.info(f"Processing: {pdf_path.name}")
        
        # Build config overrides
        config_overrides = {
            "PAGE_CAP": int(page_cap),
            "DPI": int(dpi),
            "PERFORMANCE_PRESET": performance_preset,
            "DEVICE_POLICY": device_policy
        }
        
        # Update status
        status = f"Processing {pdf_path.name}...\n"
        status += f"  Page cap: {page_cap}, DPI: {dpi}\n"
        status += f"  Preset: {performance_preset}, Device: {device_policy}\n"
        
        # Run pipeline
        result = run_pipeline(stable_path, config_overrides)
        
        if result["status"] == "error":
            error_msg = f"❌ Processing failed: {result['error']}"
            log.error(error_msg)
            return (error_msg, "", "", "", "", "", "")
        
        # Extract results
        summary = result["summary"]
        classification = result["classification"]
        
        # Load artifacts
        raw_json = ""
        normalized_json = ""
        prompt_text = ""
        
        if result["artifacts"]["raw_extraction"]:
            with open(result["artifacts"]["raw_extraction"], "r", encoding="utf-8") as f:
                raw_json = json.dumps(json.load(f), indent=2, ensure_ascii=False)
        
        if result["artifacts"]["normalized_extraction"]:
            with open(result["artifacts"]["normalized_extraction"], "r", encoding="utf-8") as f:
                normalized_json = json.dumps(json.load(f), indent=2, ensure_ascii=False)
        
        if result["artifacts"]["prompt"]:
            with open(result["artifacts"]["prompt"], "r", encoding="utf-8") as f:
                prompt_text = f.read()
        
        # Build device info
        device_info = f"Device used: {result['metadata']['device_used']}\n"
        device_info += f"Processing time: {result['metadata']['processing_time_seconds']:.2f}s\n"
        if result.get("artifacts", {}).get("debug_dir"):
            device_info += f"Debug folder: {result['artifacts']['debug_dir']}\n"
        if result['metadata']['warnings']:
            device_info += f"\n⚠️  Warnings: {'; '.join(result['metadata']['warnings'])}"
        
        # Success status
        status = f"✅ Processing complete: {pdf_path.name}\n"
        status += f"  Time: {result['metadata']['processing_time_seconds']:.2f}s\n"
        status += f"  Device: {result['metadata']['device_used']}\n"
        status += f"  Classification: {classification}"
        if result.get("artifacts", {}).get("debug_dir"):
            status += f"\n  Debug: {result['artifacts']['debug_dir']}"
        
        # Cache result
        CACHE[pdf_path.name] = result
        
        return (status, summary, classification, raw_json, normalized_json, prompt_text, device_info)
        
    except Exception as e:
        error_msg = f"❌ Unexpected error: {str(e)}"
        log.error(error_msg, exc_info=True)
        return (error_msg, "", "", "", "", "", "")


# Get default effective device
default_config = DEFAULT_CONFIG
effective_device = default_config.get_effective_device()
device_info_text = f"Default device: {effective_device}"

if effective_device == "cuda":
    import torch
    device_info_text += f" (GPU: {torch.cuda.get_device_name(0)})"

# Build UI
with gr.Blocks(title="DOTS OCR Application") as demo:
    gr.Markdown("# 📄 DOTS OCR Application")
    gr.Markdown(
        "Upload scanned PDFs for OCR extraction, analysis, and classification. "
        f"Powered by DOTS visual transformer extraction + LLM summarization "
        f"(`{MODEL}` via remote service)."
    )
    
    with gr.Row():
        file_uploader = gr.File(
            type="filepath",
            label="Upload PDF",
            file_count="single"
        )
    
    with gr.Accordion("⚙️ Processing Controls", open=True):
        with gr.Row():
            page_cap_input = gr.Number(
                label="Page Cap",
                value=DEFAULT_CONFIG.PAGE_CAP,
                precision=0,
                minimum=1,
                maximum=20,
                info="Number of pages to process"
            )
            dpi_input = gr.Number(
                label="DPI (Image Quality)",
                value=DEFAULT_CONFIG.DPI,
                precision=0,
                minimum=100,
                maximum=400,
                info="Resolution for PDF rasterization"
            )
        
        with gr.Row():
            preset_dropdown = gr.Dropdown(
                choices=["fast", "balanced", "high_quality"],
                value=DEFAULT_CONFIG.PERFORMANCE_PRESET,
                label="Performance Preset",
                info="Preset will suggest DPI and page cap values"
            )
            device_dropdown = gr.Dropdown(
                choices=["auto", "gpu", "cpu"],
                value=DEFAULT_CONFIG.DEVICE_POLICY,
                label="Device Policy",
                info="auto = use GPU if available"
            )
    
    process_button = gr.Button("🚀 Process PDF", variant="primary", size="lg")
    
    status_output = gr.Textbox(
        label="Status",
        interactive=False,
        lines=5
    )
    
    device_info_output = gr.Textbox(
        label="Device & Runtime Info",
        value=f"{device_info_text}\nLLM: {MODEL} @ {API_URL}",
        interactive=False,
        lines=3
    )
    
    gr.Markdown("## 📊 Results")
    
    with gr.Row():
        summary_output = gr.Textbox(
            label="Summary",
            lines=4,
            interactive=False
        )
        classification_output = gr.Textbox(
            label="Classification",
            interactive=False
        )
    
    with gr.Tabs():
        with gr.TabItem("Generated Prompt"):
            prompt_output = gr.Textbox(
                label="Prompt sent to Ollama",
                lines=15,
                interactive=False
            )
        
        with gr.TabItem("Normalized Extraction"):
            normalized_json_output = gr.Code(
                label="Normalized DOTS Extraction",
                language="json",
                interactive=False
            )
        
        with gr.TabItem("Raw Extraction"):
            raw_json_output = gr.Code(
                label="Raw DOTS Output",
                language="json",
                interactive=False
            )
    
    # Wire up the processing
    process_button.click(
        process_pdf_upload,
        inputs=[
            file_uploader,
            page_cap_input,
            dpi_input,
            preset_dropdown,
            device_dropdown
        ],
        outputs=[
            status_output,
            summary_output,
            classification_output,
            raw_json_output,
            normalized_json_output,
            prompt_output,
            device_info_output
        ]
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="DOTS OCR Gradio Application")
    parser.add_argument("--share", action="store_true", help="Enable Gradio share link")
    parser.add_argument("--port", type=int, default=7862, help="Gradio server port")
    args = parser.parse_args()
    
    log.info("Starting DOTS OCR Gradio application")
    log.info(f"  Output directory: {OUTPUT_DIR.absolute()}")
    log.info(f"  Default device:   {effective_device}")
    log.info(f"  LLM endpoint:     {API_URL}")
    log.info(f"  LLM model:        {MODEL}")
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
        show_error=True
    )
