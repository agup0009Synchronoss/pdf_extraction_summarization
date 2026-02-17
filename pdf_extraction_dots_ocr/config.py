"""
Configuration for DOTS OCR Application
======================================

Centralized configuration using dataclass with hardcoded defaults initially.

LLM endpoint / model / API key are NOT stored here — they live in
call_llama_api.py exactly as they do in the main pdf_extraction app.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional
import torch
import logging

log = logging.getLogger("dots_config")


@dataclass
class DotsConfig:
    """Configuration for DOTS OCR extraction and processing."""
    
    # Page selection
    PAGE_CAP: int = 2  # Default: process first 2 pages only
    
    # Rasterization quality
    DPI: int = 200  # PDF to image rendering DPI
    
    # Device policy
    DEVICE_POLICY: str = "auto"  # Options: "auto", "gpu", "cpu"
    
    # Performance presets
    PERFORMANCE_PRESET: str = "balanced"  # Options: "fast", "balanced", "high_quality"

    # vLLM server connection (local DOTS OCR model); overridable via VLLM_HOST, VLLM_PORT
    VLLM_HOST: str = field(default_factory=lambda: os.environ.get("VLLM_HOST", "localhost"))
    VLLM_PORT: int = field(default_factory=lambda: int(os.environ.get("VLLM_PORT", "8000")))
    VLLM_PROTOCOL: str = "http"
    VLLM_MODEL_NAME: str = "model"  # matches --served-model-name
    VLLM_TEMPERATURE: float = 0.1
    VLLM_TOP_P: float = 1.0
    VLLM_MAX_TOKENS: int = 16384
    DOTS_PROMPT_MODE: str = "prompt_layout_all_en"

    # NOTE: LLM endpoint / model / API key (for summary/classification) are in call_llama_api.py.

    # Classification labels (DOTS-specific)
    CLASSIFICATION_LABELS: List[str] = field(default_factory=lambda: [
        "scanned_document",
        "handwritten_form",
        "receipt",
        "invoice",
        "tax_form",
        "contract",
        "medical_record",
        "id_document",
        "financial_statement",
        "correspondence",
        "unclassified"
    ])
    
    # Prompt configuration
    MAX_PROMPT_WORDS: int = 2000  # Maximum words in Ollama prompt
    
    # Output paths
    OUTPUT_DIR: str = "output"

    # Debug mode: save per-run artifacts (page images, vLLM responses, prompt) under timestamped folder
    DEBUG_SAVE_ARTIFACTS: bool = True
    DEBUG_OUTPUT_DIR: str = "debug_output"

    def get_effective_device(self) -> str:
        """
        Determine effective device based on policy and hardware availability.
        
        Returns:
            Effective device string ("cuda" or "cpu")
        """
        if self.DEVICE_POLICY == "cpu":
            log.info("Device policy: CPU (forced)")
            return "cpu"
        
        if self.DEVICE_POLICY == "gpu":
            if torch.cuda.is_available():
                log.info("Device policy: GPU (forced), CUDA available")
                return "cuda"
            else:
                log.warning("Device policy: GPU requested but CUDA unavailable, falling back to CPU")
                return "cpu"
        
        # Auto mode
        if torch.cuda.is_available():
            log.info("Device policy: AUTO, CUDA available, using GPU")
            return "cuda"
        else:
            log.info("Device policy: AUTO, CUDA not available, using CPU")
            return "cpu"
    
    def get_preset_params(self) -> dict:
        """
        Get performance parameters based on preset.
        
        Returns:
            Dictionary with DPI, page cap, and thread parameters
        """
        presets = {
            "fast": {
                "dpi": 150,
                "page_cap": 1,
                "description": "Fast processing, lower quality"
            },
            "balanced": {
                "dpi": 200,
                "page_cap": 2,
                "description": "Balanced speed and quality"
            },
            "high_quality": {
                "dpi": 300,
                "page_cap": 5,
                "description": "High quality, slower processing"
            }
        }
        
        return presets.get(self.PERFORMANCE_PRESET, presets["balanced"])

    @property
    def vllm_base_url(self) -> str:
        """Base URL for the local vLLM server (DOTS OCR model)."""
        return f"{self.VLLM_PROTOCOL}://{self.VLLM_HOST}:{self.VLLM_PORT}/v1"


# Default global configuration
DEFAULT_CONFIG = DotsConfig()


def get_config(overrides: Optional[dict] = None) -> DotsConfig:
    """
    Get configuration with optional overrides.
    
    Args:
        overrides: Optional dictionary of config overrides
        
    Returns:
        DotsConfig instance
    """
    config = DotsConfig()
    
    if overrides:
        for key, value in overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                log.warning(f"Unknown config key: {key}")
    
    return config
