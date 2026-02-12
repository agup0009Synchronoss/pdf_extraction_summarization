"""
Configuration Management for PDF Extraction Pipeline
====================================================

Centralized configuration for all extraction parameters.
Supports runtime overrides via config_overrides dict.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class ExtractionConfig:
    """Configuration for PDF extraction and OCR processing."""
    
    # Text extraction thresholds
    MIN_WORDS: int = 200  # If text < threshold, trigger image extraction
    MAX_PROMPT_WORDS: int = 1000  # Maximum words in LLM prompt
    
    # Image extraction settings
    MAX_IMAGES: int = 5  # Maximum images to process per PDF
    MIN_IMAGE_AREA_PIXELS: int = 50_000  # Absolute minimum image size (pixels)
    MIN_IMAGE_AREA_PERCENT: float = 0.15  # Minimum 15% of estimated page area
    MIN_IMAGE_RELATIVE_TO_LARGEST: float = 0.35  # Keep major images per-page even if page estimate is noisy
    
    # Image processing
    BLIP_RESIZE: tuple = (224, 224)  # BLIP model input size
    MAX_OCR_DIMENSION: int = 2000  # Max width/height before resizing for OCR (memory safety)
    
    # OCR settings
    ENABLE_IMAGE_OCR: bool = True  # Enable/disable OCR on images
    OCR_ENGINE: str = "paddleocr"  # OCR engine: "paddleocr", "easyocr" (future)
    OCR_LANGUAGES: List[str] = field(default_factory=lambda: ["en"])  # OCR languages
    
    # Page dimension fallbacks
    DEFAULT_PAGE_WIDTH: float = 612.0  # US Letter width in PDF points
    DEFAULT_PAGE_HEIGHT: float = 792.0  # US Letter height in PDF points
    
    def apply_overrides(self, overrides: Optional[Dict[str, Any]]) -> "ExtractionConfig":
        """
        Apply runtime overrides to configuration.
        
        Args:
            overrides: Dict of config key-value pairs to override
            
        Returns:
            New ExtractionConfig with overrides applied
        """
        if not overrides:
            return self
        
        # Create a copy with overrides applied
        config_dict = {
            "MIN_WORDS": self.MIN_WORDS,
            "MAX_PROMPT_WORDS": self.MAX_PROMPT_WORDS,
            "MAX_IMAGES": self.MAX_IMAGES,
            "MIN_IMAGE_AREA_PIXELS": self.MIN_IMAGE_AREA_PIXELS,
            "MIN_IMAGE_AREA_PERCENT": self.MIN_IMAGE_AREA_PERCENT,
            "MIN_IMAGE_RELATIVE_TO_LARGEST": self.MIN_IMAGE_RELATIVE_TO_LARGEST,
            "BLIP_RESIZE": self.BLIP_RESIZE,
            "MAX_OCR_DIMENSION": self.MAX_OCR_DIMENSION,
            "ENABLE_IMAGE_OCR": self.ENABLE_IMAGE_OCR,
            "OCR_ENGINE": self.OCR_ENGINE,
            "OCR_LANGUAGES": self.OCR_LANGUAGES.copy(),
            "DEFAULT_PAGE_WIDTH": self.DEFAULT_PAGE_WIDTH,
            "DEFAULT_PAGE_HEIGHT": self.DEFAULT_PAGE_HEIGHT,
        }
        
        # Apply overrides
        for key, value in overrides.items():
            if key in config_dict:
                config_dict[key] = value
        
        return ExtractionConfig(**config_dict)


# Default global configuration
DEFAULT_CONFIG = ExtractionConfig()


def get_config(overrides: Optional[Dict[str, Any]] = None) -> ExtractionConfig:
    """
    Get configuration with optional runtime overrides.
    
    Args:
        overrides: Optional dict of config overrides
        
    Returns:
        ExtractionConfig instance
        
    Example:
        # Use defaults
        config = get_config()
        
        # Override OCR settings
        config = get_config({"ENABLE_IMAGE_OCR": False})
        
        # Override from Gradio UI
        config = get_config({
            "ENABLE_IMAGE_OCR": True,
            "OCR_LANGUAGES": ["fr"]
        })
    """
    return DEFAULT_CONFIG.apply_overrides(overrides)
