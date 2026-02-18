# DOTS OCR - Quick Start Guide

## 5-Minute Setup

### Step 1: Setup virtual environment

```bash
cd pdf_extraction_dots_ocr
```

**Windows:**
```bash
setup_venv.bat
```

**Linux/Mac:**
```bash
bash setup_venv.sh
```

### Step 2: DOTS model weights (one-time)

The DOTS model runs **in-process** via HuggingFace Transformers — no separate server needed.

**Option A: Auto-download from HuggingFace Hub (default)**

No action needed. The model downloads automatically on the first PDF upload. Requires internet access from jovyan.

**Option B: Pre-download weights locally (recommended for jovyan)**

Avoids re-downloading on every fresh environment. Run once:

```bash
git clone https://github.com/rednote-hilab/dots.ocr.git
cd dots.ocr
python3 tools/download_model.py
# Weights go to ./weights/DotsOCR
```

Then tell the app where to find them:

```bash
export DOTS_MODEL_PATH=./weights/DotsOCR
```

You can also set this in `launch.sh` or your shell profile.

### Step 3: Verify LLM API access (summary/classification)

The app uses the same remote LLM service as the main `pdf_extraction` app for summary/classification. Check `LLAMA_API_KEY` or the key in `call_llama_api.py`.

### Step 4: Test the pipeline

```bash
python test_pipeline.py
```

Expected output:
```
✅ All modules imported successfully
✅ Config loaded successfully
✅ Device detection working
✅ DotsExtractor initialized (model=rednote-hilab/dots.ocr, device: cuda/cpu)
✅ PromptBridge initialized
✅ LLM client initialized
```

### Step 5: Run the Gradio UI

```bash
bash launch.sh
# or manually:
python gradio_app.py
```

Open browser to: `http://localhost:7862`

The DOTS model loads into GPU memory on the **first PDF upload** (one-time, ~30–60s).

### Step 6: Upload and process a PDF

1. Click "Upload PDF" and select a scanned document
2. Adjust settings if needed (defaults work well):
   - Page cap: 2 pages
   - DPI: 200
   - Preset: balanced
   - Device: auto
3. Click "🚀 Process PDF"
4. View results in the tabs:
   - Summary & Classification
   - Generated Prompt
   - Normalized Extraction
   - Raw Extraction

## What Gets Created

For `sample.pdf`, you'll get:
- `output/sample_raw_dots.json` - Raw DOTS output
- `output/sample_normalized.json` - Normalized extraction
- `output/sample_prompt.txt` - Prompt sent to LLM service
- `output/sample_result.json` - Final summary + classification

Debug mode also creates `debug_output/sample_TIMESTAMP/`:
- `page_N_image.png` - Rasterized page images
- `page_N_model_response.txt` - Raw model output per page
- `page_N_model_parsed.json` - Parsed JSON per page
- `prompt_sent_to_llm.txt` - Prompt sent to Ollama
- `extraction_raw.json` / `extraction_normalized.json`

## GPU Usage

The app auto-detects and uses GPU when available:

```python
import torch
print(torch.cuda.is_available())  # Should be True
```

GPU will be shown in the UI under "Device & Runtime Info".

For faster inference with flash attention (optional):
```bash
pip install flash-attn --no-build-isolation
export DOTS_ATTN_IMPLEMENTATION=flash_attention_2
```

## Troubleshooting

**Model download fails / "Repository not found":**
- Check internet access from jovyan to `huggingface.co`
- Use Option B (pre-download) and set `DOTS_MODEL_PATH` to the local path

**"CUDA not available" / model runs on CPU:**
- App falls back to CPU automatically (slower but functional)
- Check: `python -c "import torch; print(torch.cuda.is_available())"`

**Out of GPU memory (OOM) during model load or inference:**
- Use "fast" preset (lower DPI, page cap 1)
- Reduce `DOTS_MAX_NEW_TOKENS` in `config.py`
- Ensure no other large models are loaded on the GPU

**"LLM API request failed" (summary/classification):**
- Check network access to the remote LLM service
- Verify `API_URL` / `API_KEY` in `call_llama_api.py`
- Set `LLAMA_API_KEY` env var if the key has rotated

**"ModuleNotFoundError":**
- Activate venv: `source venv_dot_ocr/bin/activate`
- Install: `pip install -r requirements.txt`

## Performance Tips

**Fast processing (higher latency on first load, then per-page):**
- Preset: "fast"
- Page cap: 1
- DPI: 150

**High quality:**
- Preset: "high_quality"
- Page cap: 5
- DPI: 300

**GPU memory issues:**
- Use "fast" preset
- Reduce page cap to 1
- Switch to CPU mode (slow but safe)
