"""
setup_model.py — One-time model download and environment check
==============================================================

Run this ONCE after cloning / pulling to pre-download the model weights to a
safe local directory (no dots in path, which avoids the transformers module
import error with rednote-hilab/dots.ocr).

Usage:
    python setup_model.py

After this script completes successfully, the app will load the model from
./hf_cache/dots_ocr on every subsequent run (no re-download needed).
"""

import os
import sys
import ssl
import urllib3
import warnings

# ── SSL bypass (mirrors gradio_app.py) ────────────────────────────────────────
os.environ['PYTHONHTTPSVERIFY'] = '0'
os.environ['REQUESTS_CA_BUNDLE'] = ''
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['SSL_CERT_FILE'] = ''
os.environ['HF_HUB_DISABLE_SSL_VERIFICATION'] = '1'
os.environ['HF_HOME'] = './hf_cache'

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings('ignore', message='Unverified HTTPS request')
ssl._create_default_https_context = ssl._create_unverified_context

try:
    import requests
    class _UnverifiedSession(requests.Session):
        def __init__(self):
            super().__init__()
            self.verify = False
    requests.Session = _UnverifiedSession
except ImportError:
    pass

try:
    import httpx
    _orig = httpx.Client.__init__
    def _patched(self, *a, **kw): kw['verify'] = False; _orig(self, *a, **kw)
    httpx.Client.__init__ = _patched
    _orig_a = httpx.AsyncClient.__init__
    def _patched_a(self, *a, **kw): kw['verify'] = False; _orig_a(self, *a, **kw)
    httpx.AsyncClient.__init__ = _patched_a
except ImportError:
    pass

try:
    from huggingface_hub import configure_http_backend
    import requests as _req
    def _backend():
        s = _req.Session(); s.verify = False; return s
    configure_http_backend(backend_factory=_backend)
except Exception:
    pass
# ──────────────────────────────────────────────────────────────────────────────

from pathlib import Path

HF_MODEL_ID = "rednote-hilab/dots.ocr"
LOCAL_DIR    = Path("./hf_cache/dots_ocr")


def check_dependencies():
    print("─" * 60)
    print("Checking dependencies...")
    ok = True

    packages = {
        "torch":           "torch",
        "torchvision":     "torchvision",
        "transformers":    "transformers",
        "accelerate":      "accelerate",
        "huggingface_hub": "huggingface_hub",
        "qwen_vl_utils":   "qwen_vl_utils",
        "fitz (PyMuPDF)":  "fitz",
        "PIL (Pillow)":    "PIL",
    }

    for label, module in packages.items():
        try:
            m = __import__(module)
            ver = getattr(m, "__version__", "?")
            print(f"  ✅ {label}: {ver}")
        except ImportError:
            print(f"  ❌ {label}: NOT FOUND  ← install with: pip install {module.split('.')[0]}")
            ok = False

    try:
        import torch
        cuda_ok = torch.cuda.is_available()
        if cuda_ok:
            print(f"  ✅ CUDA: available ({torch.cuda.get_device_name(0)})")
        else:
            print("  ⚠️  CUDA: not available — model will run on CPU (very slow)")
    except Exception as e:
        print(f"  ⚠️  CUDA check failed: {e}")

    return ok


def download_model():
    print("─" * 60)
    if (LOCAL_DIR / "config.json").exists():
        print(f"✅ Model already downloaded at: {LOCAL_DIR.resolve()}")
    else:
        print(f"Downloading {HF_MODEL_ID} → {LOCAL_DIR.resolve()}")
        print("(This is ~6 GB and only happens once)\n")
        LOCAL_DIR.mkdir(parents=True, exist_ok=True)

        try:
            from huggingface_hub import snapshot_download
            snapshot_download(HF_MODEL_ID, local_dir=str(LOCAL_DIR))
            print(f"\n✅ Model downloaded to: {LOCAL_DIR.resolve()}")
        except Exception as e:
            print(f"\n❌ Download failed: {e}")
            return False

    patch_model_config()
    return True


def patch_model_config():
    """
    Apply the video_processor fix from HF dots.ocr discussion #38.

    The upstream model's configuration_dots.py does not override `attributes`
    on DotsVLProcessor, so the parent class (Qwen2_5_VLProcessor) injects a
    `video_processor` requirement that fails with NoneType.

    Fix: override `attributes` to only list image_processor and tokenizer,
    and accept (but ignore) video_processor=None in __init__.
    """
    config_file = LOCAL_DIR / "configuration_dots.py"
    if not config_file.exists():
        print("  ⚠️  configuration_dots.py not found — skipping patch")
        return

    content = config_file.read_text(encoding="utf-8")

    if 'attributes = ["image_processor", "tokenizer"]' in content:
        print("  ✅ video_processor patch already applied")
        return

    print("  Applying video_processor patch to configuration_dots.py ...")

    content = content.replace(
        "class DotsVLProcessor(Qwen2_5_VLProcessor):\n"
        "    def __init__(self, image_processor=None, tokenizer=None, chat_template=None, **kwargs):",
        "class DotsVLProcessor(Qwen2_5_VLProcessor):\n"
        '    attributes = ["image_processor", "tokenizer"]\n'
        "    def __init__(self, image_processor=None, tokenizer=None, video_processor=None, chat_template=None, **kwargs):",
    )

    config_file.write_text(content, encoding="utf-8")
    print("  ✅ Patch applied successfully")


def verify_model_loads():
    print("─" * 60)
    print("Verifying model loads correctly...")
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        print(f"  Loading processor from {LOCAL_DIR} ...")
        processor = AutoProcessor.from_pretrained(
            str(LOCAL_DIR),
            trust_remote_code=True,
        )
        print("  ✅ Processor loaded")

        print(f"  Loading model (bfloat16, device_map=auto) ...")
        model = AutoModelForCausalLM.from_pretrained(
            str(LOCAL_DIR),
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        print(f"  ✅ Model loaded on: {next(model.parameters()).device}")

        del model, processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("✅ Verification complete — ready to run gradio_app.py")
        return True
    except Exception as e:
        print(f"❌ Model load failed: {e}")
        return False


def print_env_hint():
    print("─" * 60)
    print("To tell the app to use the pre-downloaded local model, either:")
    print(f"  export DOTS_MODEL_PATH={LOCAL_DIR.resolve()}")
    print("  (or it will be auto-resolved by _resolve_model_path() in extractor.py)")
    print()
    print("To enable flash attention (if installed):")
    print("  pip install flash-attn --no-build-isolation")
    print("  export DOTS_ATTN_IMPLEMENTATION=flash_attention_2")
    print("─" * 60)


if __name__ == "__main__":
    print("\n🔧 DOTS OCR — Setup & Model Download\n")

    deps_ok = check_dependencies()
    if not deps_ok:
        print("\n❌ Please install missing packages first, then re-run this script.")
        sys.exit(1)

    dl_ok = download_model()
    if not dl_ok:
        sys.exit(1)

    verify_ok = verify_model_loads()
    print_env_hint()

    sys.exit(0 if verify_ok else 1)
