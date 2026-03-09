"""
SSL Bypass wrapper for running Docling CLI
===========================================

This script applies full SSL bypass monkey-patching before invoking the Docling
CLI, so HuggingFace model downloads work in corporate environments with SSL
inspection.

Usage:
    python run_docling_ssl_bypass.py --pipeline vlm --vlm-model granite_docling --to md --output ./out "file.pdf"
"""
# ---------------------------------------------------------------------------
# SSL bypass — MUST run before any other imports
import os
import ssl
import urllib3
import warnings

os.environ['PYTHONHTTPSVERIFY'] = '0'
os.environ['REQUESTS_CA_BUNDLE'] = ''
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['SSL_CERT_FILE'] = ''
os.environ['HF_HUB_DISABLE_SSL_VERIFICATION'] = '1'
os.environ['HF_HOME'] = './hf_cache'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
# Disable HuggingFace XET (Rust-level large-file downloader) — it ignores
# Python SSL patches and panics on corporate networks with SSL inspection.
os.environ['HF_HUB_DISABLE_XET'] = '1'

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings('ignore', message='Unverified HTTPS request')

ssl._create_default_https_context = ssl._create_unverified_context

# Patch requests BEFORE importing anything that uses it
import requests

class _UnverifiedSession(requests.Session):
    def __init__(self):
        super().__init__()
        self.verify = False

requests.Session = _UnverifiedSession

# Patch httpx BEFORE huggingface_hub loads
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

# Patch huggingface_hub's session factory
try:
    from huggingface_hub import configure_http_backend

    def _unverified_backend() -> requests.Session:
        session = requests.Session()
        session.verify = False
        return session

    configure_http_backend(backend_factory=_unverified_backend)
except Exception:
    pass
# ---------------------------------------------------------------------------

# NOW we can import docling's CLI and run it
import sys
from docling.cli.main import app as docling_cli_app

if __name__ == "__main__":
    print(f"Running Docling with SSL bypass patches applied")
    print(f"Command: docling {' '.join(sys.argv[1:])}")
    print()
    
    # Invoke the Typer app directly (this is what the docling CLI entry point does)
    docling_cli_app()
