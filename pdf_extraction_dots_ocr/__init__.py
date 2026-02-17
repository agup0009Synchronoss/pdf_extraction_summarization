"""
DOTS OCR Application Package
============================

Scanned PDF OCR extraction using DOTS visual transformer + Ollama summarization.
"""

__version__ = "0.1.0"

from .config import DotsConfig, get_config, DEFAULT_CONFIG
from .extractor import DotsExtractor, extract_pdf_with_dots
from .prompt_bridge import PromptBridge, build_ollama_prompt
from .ollama_client import OllamaClient, call_ollama
from .pipeline import DotsPipeline, run_pipeline

__all__ = [
    "DotsConfig",
    "get_config",
    "DEFAULT_CONFIG",
    "DotsExtractor",
    "extract_pdf_with_dots",
    "PromptBridge",
    "build_ollama_prompt",
    "OllamaClient",
    "call_ollama",
    "DotsPipeline",
    "run_pipeline",
]
