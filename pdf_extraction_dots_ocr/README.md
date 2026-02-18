# DOTS OCR Application

Separate application for scanned PDF OCR extraction using DOTS visual transformer +
LLM summarization via the same remote service used by the main `pdf_extraction` app.

## Architecture

This application is fully isolated from the main `pdf_extraction` app and follows a clean pipeline:

1. **DOTS Extraction** - Extract OCR/layout from scanned PDF pages using HuggingFace Transformers (in-process, no server)
2. **Normalization** - Convert raw DOTS output to stable schema with lenient parsing
3. **Prompt Bridge** - Transform normalized extraction into a custom LLM prompt
4. **LLM Inference** - Generate summary and classification via remote service (fail-fast, no fallback)
5. **Persistence** - Save all artifacts for debugging and analysis

## Quick start (Linux / jovyan)

One command starts the Gradio app. The DOTS model loads in-process on the first PDF upload.

```bash
cd pdf_extraction_dots_ocr
bash setup_venv.sh          # first time only
bash launch.sh
```

Then open `http://localhost:7862`.

Optional env vars: `DOTS_MODEL_PATH` (default `rednote-hilab/dots.ocr`), `GRADIO_PORT` (default 7862).

## Setup

### 1. Create dedicated virtual environment

**Linux:** use the setup script (recommended):

```bash
cd pdf_extraction_dots_ocr
bash setup_venv.sh
```

**Manual (any OS):**

```bash
cd pdf_extraction_dots_ocr
python -m venv venv_dot_ocr
```

### 2. Activate environment

**Windows:**
```bash
venv_dot_ocr\Scripts\activate
```

**Linux/Mac:**
```bash
source venv_dot_ocr/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. DOTS model (HuggingFace Transformers — in-process)

The extractor loads `rednote-hilab/dots.ocr` directly in-process via Transformers. No separate server is needed.

**Option A: Auto-download from HuggingFace Hub (default)**

No action needed. The model downloads on the first PDF upload. Requires internet from jovyan.

**Option B: Pre-download weights locally (recommended for jovyan)**

```bash
git clone https://github.com/rednote-hilab/dots.ocr.git
cd dots.ocr
python3 tools/download_model.py
# Weights go to ./weights/DotsOCR (directory name without periods is required)
```

Point the app to your weights:
```bash
export DOTS_MODEL_PATH=./weights/DotsOCR
```

Or set it permanently in `launch.sh`.

**Optional: faster inference with flash attention**

```bash
pip install flash-attn --no-build-isolation
export DOTS_ATTN_IMPLEMENTATION=flash_attention_2
```

Without `flash-attn`, the app uses `sdpa` (PyTorch's built-in, safe default).

### 5. LLM API access (summary/classification)

This app calls the same remote LLM cron-job service used by the main `pdf_extraction` app.
The endpoint, model, and API key are configured in `call_llama_api.py` — no local Ollama required.

To update the key or endpoint, edit `call_llama_api.py` (or set the `LLAMA_API_KEY` environment variable).

## Usage

### Run Gradio UI

```bash
bash launch.sh
# or manually (after activating venv):
python gradio_app.py
```

Then open browser to `http://localhost:7862` (or `GRADIO_PORT` if set).

### Run from command line (pipeline)

```python
from pathlib import Path
from pipeline import run_pipeline

result = run_pipeline(Path("sample.pdf"))
print(result["summary"])
print(result["classification"])
```

## Configuration

Edit `config.py` or pass overrides:

```python
config_overrides = {
    "PAGE_CAP": 5,
    "DPI": 300,
    "DEVICE_POLICY": "gpu",
    "PERFORMANCE_PRESET": "high_quality",
    "DOTS_MODEL_PATH": "./weights/DotsOCR",
}

result = run_pipeline(pdf_path, config_overrides)
```

Key config fields:

| Field | Default | Description |
|---|---|---|
| `DOTS_MODEL_PATH` | `rednote-hilab/dots.ocr` | HF Hub ID or local path to model weights |
| `DOTS_MAX_NEW_TOKENS` | `24000` | Max tokens for model.generate() |
| `DOTS_ATTN_IMPLEMENTATION` | `sdpa` | Attention backend; use `flash_attention_2` with flash-attn installed |
| `PAGE_CAP` | `2` | Max pages to process per PDF |
| `DPI` | `200` | PDF rasterization resolution |
| `DEVICE_POLICY` | `auto` | `auto`, `gpu`, or `cpu` |

## Key Features

- **In-process inference**: DOTS model runs directly in Python via Transformers — no external server required
- **GPU Auto-Detection**: Automatically uses GPU when available, with CPU fallback
- **Page Cap Control**: Limit pages processed for cost/latency management (default: 2 pages)
- **Quality Presets**: Fast, balanced, high_quality presets adjust DPI and page caps
- **Lenient Parsing**: Handles DOTS response format drift with warnings
- **Full Artifact Persistence**: Saves raw, normalized, prompt, and response for every run
- **Fail-Fast**: Clear error messages, no silent fallbacks on critical paths
- **Debug Mode (default on)**: Per-run timestamped folder under `debug_output/` with page images, model responses, and prompt for easier debugging

## Debug mode

When `DEBUG_SAVE_ARTIFACTS` is true (default), each run creates a folder `debug_output/{filename}_{timestamp}/` containing:

- `page_1_image.png`, `page_2_image.png`, ... — rasterized page images sent to the model
- `page_1_model_response.txt`, ... — raw model response text per page
- `page_1_model_parsed.json`, ... — parsed layout/OCR JSON per page
- `prompt_sent_to_llm.txt` — the prompt built from extraction and sent to the summary/classification LLM
- `extraction_raw.json` — full raw extraction
- `extraction_normalized.json` — normalized extraction

Example: for `vaccination.pdf` at 14:30:52 you get `debug_output/vaccination_20260217_143052/`. Turn off by setting `DEBUG_SAVE_ARTIFACTS = False` in `config.py` or via overrides.

## Output Files

For each PDF processed, the following files are created in `output/`:

- `{name}_raw_dots.json` - Raw DOTS extraction output
- `{name}_normalized.json` - Normalized extraction in stable schema
- `{name}_prompt.txt` - Generated prompt sent to the LLM service
- `{name}_result.json` - Complete result with summary, classification, and metadata

## Device Policy

- `auto` (default): Use GPU if CUDA available, otherwise CPU
- `gpu`: Force GPU (fails if unavailable)
- `cpu`: Force CPU

Device used is logged and included in output metadata.

## Classification Labels

DOTS-specific labels:
- scanned_document
- handwritten_form
- receipt
- invoice
- tax_form
- contract
- medical_record
- id_document
- financial_statement
- correspondence
- unclassified

## Backend Status

**Current**: Real DOTS OCR extraction via HuggingFace Transformers (in-process). PDF pages are rasterized with PyMuPDF, resized to model-friendly bounds, and fed directly to `model.generate()` using the Transformers library. The response is parsed and normalized for the prompt bridge. Summary and classification use the remote LLM cron-job service (`call_llama_api.py`).

## Differences from Main App

This DOTS OCR app differs from the main `pdf_extraction` app:

1. **Separate stack**: Own venv, dependencies, config
2. **DOTS extraction only**: No Docling, no embedded-image OCR
3. **Page rasterization**: Full-page image approach instead of text-first
4. **New labels**: DOTS-specific classification schema
5. **Fail-fast**: No fallback paths for clarity
6. **GPU-first**: Automatic GPU usage with telemetry

## Troubleshooting

**GPU not detected:**
- Check CUDA installation: `python -c "import torch; print(torch.cuda.is_available())"`
- Verify PyTorch CUDA build: `python -c "import torch; print(torch.version.cuda)"`

**Out of GPU memory (OOM):**
- Reduce DPI or use "fast" preset
- Reduce page cap
- Switch to CPU mode if GPU memory insufficient

**Model download fails / repository not found:**
- Check internet access to `huggingface.co` from jovyan
- Use local weights (Option B in Setup Step 4) and set `DOTS_MODEL_PATH`

**"LLM API request failed" / connection error:**
- Check network access to the remote LLM service
- Verify `API_URL` and `API_KEY` in `call_llama_api.py`
- Set the `LLAMA_API_KEY` environment variable if the key has rotated

**Empty extraction JSON (extraction_raw.json / extraction_normalized.json):**
- Check the `page_N_model_response.txt` files in the debug folder to see raw model output
- If they are empty, model inference failed — check terminal logs for errors
- GPU OOM is a common cause; try the "fast" preset or CPU mode
