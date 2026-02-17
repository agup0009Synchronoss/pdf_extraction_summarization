"""
OCR Handler Module
==================

Abstracts OCR functionality to support multiple engines.
Currently supports PaddleOCR with extensible design for future engines (EasyOCR, Tesseract, etc.).
"""

from __future__ import annotations
import logging
from typing import List, Optional
from PIL import Image

# ---------------------------------------------------------------------------
# Setup logging
log = logging.getLogger("ocr_handler")

# Supported language mappings (UI-friendly -> engine-specific)
_PADDLE_LANG_MAP = {
    "en": "en",
    "ch": "ch",
    "fr": "french",
    "de": "german",
    "es": "es",
    "it": "it",
    "ja": "japan",
    "ko": "korean",
    "ru": "cyrillic",
}

# ---------------------------------------------------------------------------
# PaddleOCR (optional dependency)
try:
    from paddleocr import PaddleOCR
    
    # Initialize PaddleOCR with common settings
    # Note: First run downloads models (~50MB)
    _PADDLE_OCR_INSTANCES = {}  # Cache by language
    PADDLE_OK = True
    log.info(" PaddleOCR available")
except ImportError as e:
    PADDLE_OK = False
    _PADDLE_OCR_INSTANCES = {}
    log.warning(f" PaddleOCR not available: {e}")

# ---------------------------------------------------------------------------
# EasyOCR (optional fallback - already in requirements.txt)
try:
    import easyocr
    _EASY_OCR_READER = None  # Lazy initialization
    EASY_OK = True
    log.info(" EasyOCR available")
except ImportError:
    EASY_OK = False
    _EASY_OCR_READER = None
    log.warning(" EasyOCR not available")


# ---------------------------------------------------------------------------
def _get_paddleocr_instance(lang: str) -> Optional[PaddleOCR]:
    """
    Get or create PaddleOCR instance for given language.
    Instances are cached to avoid re-initialization overhead.
    """
    if not PADDLE_OK:
        return None
    
    if lang not in _PADDLE_OCR_INSTANCES:
        try:
            log.info(f"Initializing PaddleOCR for language: {lang}")
            # Use minimal parameters for maximum compatibility
            _PADDLE_OCR_INSTANCES[lang] = PaddleOCR(
                use_angle_cls=True,
                lang=lang
            )
        except Exception as e:
            log.error(f"Failed to initialize PaddleOCR: {e}")
            return None
    
    return _PADDLE_OCR_INSTANCES[lang]


def _paddleocr_extract(image: Image.Image, languages: List[str]) -> str:
    """
    Extract text from image using PaddleOCR.
    
    Args:
        image: PIL Image
        languages: List of language codes (uses first one)
        
    Returns:
        Extracted text as string
    """
    raw_lang = languages[0] if languages else "en"
    lang = _PADDLE_LANG_MAP.get(raw_lang, "en")
    ocr_instance = _get_paddleocr_instance(lang)
    
    if not ocr_instance:
        return ""
    
    try:
        # PaddleOCR expects numpy array or file path
        import numpy as np
        img_array = np.array(image)
        
        # Run OCR
        result = ocr_instance.ocr(img_array, cls=True)
        
        if not result or not result[0]:
            return ""
        
        # Extract text from results
        # PaddleOCR returns: [[[bbox], (text, confidence)], ...]
        lines = []
        for line in result[0]:
            if line and len(line) >= 2:
                text = line[1][0] if isinstance(line[1], tuple) else str(line[1])
                lines.append(text)
        
        extracted_text = "\n".join(lines)
        log.info(f"PaddleOCR extracted {len(extracted_text)} characters")
        return extracted_text
        
    except Exception as e:
        log.warning(f"PaddleOCR extraction failed: {e}")
        return ""


def _easyocr_extract(image: Image.Image, languages: List[str]) -> str:
    """
    Extract text from image using EasyOCR (fallback).
    
    Args:
        image: PIL Image
        languages: List of language codes
        
    Returns:
        Extracted text as string
    """
    global _EASY_OCR_READER
    
    if not EASY_OK:
        return ""
    
    try:
        # Lazy initialization
        if _EASY_OCR_READER is None:
            lang_list = languages if languages else ["en"]
            log.info(f"Initializing EasyOCR for languages: {lang_list}")
            _EASY_OCR_READER = easyocr.Reader(lang_list, gpu=False)
        
        # Convert PIL to numpy
        import numpy as np
        img_array = np.array(image)
        
        # Run OCR
        result = _EASY_OCR_READER.readtext(img_array, detail=0)  # detail=0 returns text only
        
        if not result:
            return ""
        
        extracted_text = "\n".join(result)
        log.info(f"EasyOCR extracted {len(extracted_text)} characters")
        return extracted_text
        
    except Exception as e:
        log.warning(f"EasyOCR extraction failed: {e}")
        return ""


# ---------------------------------------------------------------------------
def extract_text_from_image(
    image: Image.Image,
    engine: str = "paddleocr",
    languages: Optional[List[str]] = None
) -> str:
    """
    Extract text from image using specified OCR engine.
    
    Args:
        image: PIL Image to extract text from
        engine: OCR engine to use ("paddleocr", "easyocr")
        languages: List of language codes (e.g., ["en"], ["ch"], ["fr"])
        
    Returns:
        Extracted text as string, or empty string if OCR fails/unavailable
        
    Example:
        >>> from PIL import Image
        >>> img = Image.open("document.jpg")
        >>> text = extract_text_from_image(img, engine="paddleocr", languages=["en"])
        >>> print(text)
    """
    if languages is None:
        languages = ["en"]
    
    if not image:
        return ""
    
    # Try primary engine
    if engine == "paddleocr" and PADDLE_OK:
        result = _paddleocr_extract(image, languages)
        if result:
            return result
        # Fall through to EasyOCR if PaddleOCR returns empty
    
    if engine == "easyocr" and EASY_OK:
        return _easyocr_extract(image, languages)
    
    # Fallback: try EasyOCR if PaddleOCR was requested but failed/unavailable
    if engine == "paddleocr" and EASY_OK:
        log.info("PaddleOCR unavailable, falling back to EasyOCR")
        return _easyocr_extract(image, languages)
    
    # No OCR available
    if engine == "paddleocr" and not PADDLE_OK:
        log.warning("PaddleOCR not available. Install: pip install paddleocr paddlepaddle")
    elif engine == "easyocr" and not EASY_OK:
        log.warning("EasyOCR not available. Install: pip install easyocr")
    
    return ""


def is_ocr_available(engine: str = "paddleocr") -> bool:
    """
    Check if specified OCR engine is available.
    
    Args:
        engine: OCR engine name
        
    Returns:
        True if engine is available, False otherwise
    """
    if engine == "paddleocr":
        return PADDLE_OK
    elif engine == "easyocr":
        return EASY_OK
    return False
