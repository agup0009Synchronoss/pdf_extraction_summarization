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
python -m venv venv_dot_ocr
source venv_dot_ocr/bin/activate
pip install -r requirements.txt
```

### Step 2: Start the DOTS vLLM server (local OCR model)

Real extraction needs a **local** vLLM server running the DOTS OCR model. One-time setup:

1. Clone and install DOTS OCR (if not already done):

   ```bash
   git clone https://github.com/rednote-hilab/dots.ocr.git
   cd dots.ocr
   pip install -e .
   ```

2. Download weights:

   ```bash
   python3 tools/download_model.py
   ```

3. Start the server (GPU with ~6GB+ VRAM):

   ```bash
   CUDA_VISIBLE_DEVICES=0 vllm serve rednote-hilab/dots.ocr-1.5 --tensor-parallel-size 1 --gpu-memory-utilization 0.9 --chat-template-content-format string --served-model-name model --trust-remote-code
   ```

   Leave this running. Default URL: `http://localhost:8000/v1`.

### Step 3: Verify LLM API access (summary/classification)

The app uses the same remote LLM service as the main `pdf_extraction` app for summary/classification. Check `LLAMA_API_KEY` or the key in `call_llama_api.py`.

### Step 4: Test the pipeline

```bash
python test_pipeline.py
```

Expected output (vLLM server must be running for full extraction):
```
✅ All modules imported successfully
✅ Config loaded successfully
✅ Device detection working
✅ DotsExtractor initialized (vLLM at http://localhost:8000/v1, device: cuda/cpu)
✅ PromptBridge initialized
✅ LLM client initialized
```

### Step 5: Run the Gradio UI

```bash
python gradio_app.py
```

Open browser to: `http://localhost:7862`

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

## GPU Usage

The app auto-detects and uses GPU when available:

Check if GPU is detected:
```python
import torch
print(torch.cuda.is_available())  # Should be True
```

GPU will be shown in the UI under "Device & Runtime Info"

## Troubleshooting

**vLLM connection refused (extraction fails):**
- Start the DOTS vLLM server (Step 2). Ensure it is listening at `http://localhost:8000/v1`.
- Check `config.py`: `VLLM_HOST`, `VLLM_PORT`.

**"LLM API request failed" (summary/classification):**
- Check network access to the remote LLM service
- Verify `API_URL` / `API_KEY` in `call_llama_api.py`
- Set `LLAMA_API_KEY` env var if the key has rotated

**"CUDA not available"**
- App will fallback to CPU automatically
- Check: `python -c "import torch; print(torch.cuda.is_available())"`

**"ModuleNotFoundError"**
- Activate venv: `venv_dot_ocr\Scripts\activate`
- Install: `pip install -r requirements.txt`

## Next Steps

- Edit `config.py` to customize defaults (e.g. `VLLM_HOST`, `VLLM_PORT`, `PAGE_CAP`, `DPI`)
- Add your own classification labels
- Run vLLM on another machine and set `VLLM_HOST` to that host

## Performance Tips

**Fast processing (1-2 seconds per page):**
- Preset: "fast"
- Page cap: 1
- DPI: 150

**High quality (5-10 seconds per page):**
- Preset: "high_quality"
- Page cap: 5
- DPI: 300

**GPU memory issues:**
- Use "fast" preset
- Reduce page cap to 1
- Switch to CPU mode
