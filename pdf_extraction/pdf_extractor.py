"""
PDF EXTRACTOR
=============

Single-file implementation of:
 • Docling text extraction  (embedded-text + OCR)
 • Optional pikepdf image extraction + BLIP captioning
 • Image OCR extraction (PaddleOCR/EasyOCR)
 • Outputs one <stem>_dual.json next to the source or to --output-dir
"""

from __future__ import annotations
import base64, datetime, io, json, logging, sys, tempfile, time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pikepdf
from pikepdf import PdfImage
from PIL import Image

# Import config and OCR handler
from config import get_config, ExtractionConfig
from ocr_handler import extract_text_from_image, is_ocr_available

# ---------------------------------------------------------------------------
# OPTIONAL deep-learning deps (handled gracefully if missing)
try:
    from transformers import BlipProcessor, BlipForConditionalGeneration   # type: ignore
    import torch                                                          # type: ignore

    _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    _BLIP_PROCESSOR = BlipProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )
    _BLIP_MODEL = (
        BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )
        .to(_DEVICE)
        .eval()
    )
    BLIP_OK = True
except Exception:
    BLIP_OK = False
    _BLIP_PROCESSOR = _BLIP_MODEL = _DEVICE = None  # type: ignore

# ---------------------------------------------------------------------------
# DOC LING
try:
    from docling.document_converter import DocumentConverter               # type: ignore
    from docling.datamodel.document import ConversionResult                # type: ignore
except ImportError as exc:  # pragma: no cover
    sys.exit(
        "❌  Docling not installed.  pip install docling   "
        f"(Import error: {exc!s})"
    )

# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("pdf_extractor")

# ---------------------------------------------------------------------------
# Helper function to resize image keeping aspect ratio
def _resize_keeping_aspect(img: Image.Image, max_dimension: int) -> Image.Image:
    """Resize image so largest dimension is max_dimension, keeping aspect ratio."""
    w, h = img.size
    if max(w, h) <= max_dimension:
        return img
    
    if w > h:
        new_w = max_dimension
        new_h = int(h * max_dimension / w)
    else:
        new_h = max_dimension
        new_w = int(w * max_dimension / h)
    
    return img.resize((new_w, new_h), Image.Resampling.LANCZOS)

# ---------------------------------------------------------------------------
# Text cleaning function to handle OCR artifacts and Unicode issues
def clean_text(text: str) -> str:
    """Clean OCR text by removing problematic Unicode characters and artifacts."""
    if not text:
        return text
    
    # Remove or replace problematic Unicode characters
    import re
    
    # Replace common problematic Unicode characters with ASCII equivalents
    replacements = {
        '\u01af': 'U',  # Latin capital letter U with horn
        '\u01b0': 'u',  # Latin small letter u with horn
        '\u2217': '*',  # Asterisk operator
        '\u2013': '-',  # En dash
        '\u2014': '--', # Em dash
        '\u2018': "'",  # Left single quotation mark
        '\u2019': "'",  # Right single quotation mark
        '\u201c': '"',  # Left double quotation mark
        '\u201d': '"',  # Right double quotation mark
        '\u2022': '•',  # Bullet
        '\u00a0': ' ',  # Non-breaking space
        '\u00b0': '°',  # Degree sign
        '\u00ae': '®',  # Registered trademark
        '\u00a9': '©',  # Copyright
        '\u2122': '™',  # Trademark
    }
    
    cleaned = text
    for unicode_char, replacement in replacements.items():
        cleaned = cleaned.replace(unicode_char, replacement)
    
    # Remove other non-printable characters except basic whitespace
    cleaned = re.sub(r'[^\x20-\x7E\n\r\t]', '', cleaned)
    
    # Normalize whitespace
    cleaned = re.sub(r'\s+', ' ', cleaned)
    cleaned = cleaned.strip()
    
    return cleaned

# ---------------------------------------------------------------------------

def _caption(pil: Image.Image) -> str:
    """Return BLIP caption or placeholder."""
    if not BLIP_OK:
        return "<BLIP unavailable>"
    try:
        inputs = _BLIP_PROCESSOR(images=pil, return_tensors="pt").to(_DEVICE)
        out = _BLIP_MODEL.generate(**inputs)
        caption = _BLIP_PROCESSOR.decode(out[0], skip_special_tokens=True).strip()
        return clean_text(caption)
    except Exception as e:                        # pragma: no cover
        log.warning("BLIP failed: %s", e)
        return "<caption error>"

def _extract_images(pdf_path: Path, config: ExtractionConfig) -> List[Dict[str, Any]]:
    """
    Return image elements with captions and OCR text (largest by pixel area).

    Robustness notes:
    - Filters with an absolute pixel floor.
    - Filters with relative coverage against estimated page area.
    - Keeps significant images per page using a ratio to the largest page image,
      so useful images survive even when page-area estimation is imperfect.
    """
    pdf = pikepdf.open(pdf_path)
    elements: List[Dict[str, Any]] = []

    ocr_enabled = config.ENABLE_IMAGE_OCR and is_ocr_available(config.OCR_ENGINE)
    if config.ENABLE_IMAGE_OCR and not ocr_enabled:
        log.warning("OCR requested but engine '%s' is unavailable; image OCR disabled", config.OCR_ENGINE)

    all_candidates: List[Dict[str, Any]] = []

    for p_idx, page in enumerate(pdf.pages, 1):
        # Estimate page area in point-space; treated as a stable proxy for relative filtering.
        try:
            box = page.mediabox if hasattr(page, "mediabox") else page.MediaBox
            page_width = float(box[2] - box[0])
            page_height = float(box[3] - box[1])
            page_area_est = max(page_width * page_height, 1.0)
        except Exception as exc:
            log.warning("Failed to get page dimensions, using defaults: %s", exc)
            page_area_est = config.DEFAULT_PAGE_WIDTH * config.DEFAULT_PAGE_HEIGHT

        page_candidates: List[Dict[str, Any]] = []

        for name, obj in page.images.items():
            img = PdfImage(obj)
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                out = img.extract_to(fileprefix=tmp.name.rstrip(".png"))
            try:
                pil = Image.open(out).convert("RGB")
            finally:
                Path(out).unlink(missing_ok=True)

            w, h = pil.size
            image_area = w * h

            if image_area < config.MIN_IMAGE_AREA_PIXELS:
                continue

            page_candidates.append(
                {
                    "pil": pil,
                    "page": p_idx,
                    "name": name,
                    "area": image_area,
                    "page_area_est": page_area_est,
                    "image_width": w,
                    "image_height": h,
                }
            )

        if not page_candidates:
            continue

        largest_area = max(c["area"] for c in page_candidates)

        for candidate in page_candidates:
            area = candidate["area"]
            page_area_est = candidate["page_area_est"]
            page_coverage = area / page_area_est if page_area_est > 0 else 0.0
            rel_to_largest = area / largest_area if largest_area > 0 else 0.0

            # Keep if it is page-significant OR among major images on that page.
            if (
                page_coverage >= config.MIN_IMAGE_AREA_PERCENT
                or rel_to_largest >= config.MIN_IMAGE_RELATIVE_TO_LARGEST
            ):
                candidate["page_coverage"] = page_coverage
                candidate["relative_to_largest"] = rel_to_largest
                all_candidates.append(candidate)

    if not all_candidates:
        return []

    all_candidates.sort(key=lambda m: m["area"], reverse=True)
    selected = all_candidates[: config.MAX_IMAGES]

    for idx, m in enumerate(selected, 1):
        pil = m["pil"]

        # BLIP caption path
        pil224 = pil.resize(config.BLIP_RESIZE)
        caption = _caption(pil224)

        # OCR path
        ocr_text = ""
        ocr_engine_used = None
        if ocr_enabled:
            ocr_pil = pil
            if max(pil.size) > config.MAX_OCR_DIMENSION:
                log.info("Resizing image from %s for OCR (max: %s)", pil.size, config.MAX_OCR_DIMENSION)
                ocr_pil = _resize_keeping_aspect(pil, config.MAX_OCR_DIMENSION)

            ocr_text = clean_text(
                extract_text_from_image(
                    ocr_pil,
                    engine=config.OCR_ENGINE,
                    languages=config.OCR_LANGUAGES,
                )
            )
            ocr_engine_used = config.OCR_ENGINE if ocr_text else None

        buffer = io.BytesIO()
        pil224.save(buffer, format="JPEG", quality=85)

        elements.append(
            {
                "element_id": f"image_{idx}",
                "element_type": "image",
                "page_number": m["page"],
                "order": 0,
                "caption": caption,
                "ocr_text": ocr_text,
                "image_224_jpeg_base64": base64.b64encode(buffer.getvalue()).decode(),
                "metadata": {
                    "image_width": m["image_width"],
                    "image_height": m["image_height"],
                    "page_area_percentage_est": round(m["page_coverage"] * 100, 2),
                    "relative_to_largest_on_page": round(m["relative_to_largest"], 3),
                    "ocr_engine": ocr_engine_used,
                },
            }
        )

    return elements

# ---------------------------------------------------------------------------

def _docling_extract(path: Path) -> tuple[list[Dict[str, Any]], int]:
    """Return text elements + page count using Docling converter."""
    try:
        # Use the working pattern from oldcode
        converter = DocumentConverter()
        result = converter.convert(str(path))
        
        if not result or not result.document:
            raise Exception(f"Docling failed to process document: {path}")
        
        doc = result.document
        page_cnt = 1  # Default fallback
        
        # Try to get page count
        if hasattr(result, 'pages') and result.pages:
            page_cnt = len(result.pages)
        elif hasattr(doc, 'pages') and doc.pages:
            page_cnt = len(doc.pages)
        elif hasattr(doc, 'page_count'):
            page_cnt = doc.page_count
        
        text_elements = []
        
        # Use Docling's export_to_text method (from working code)
        if hasattr(doc, 'export_to_text'):
            full_text = doc.export_to_text()
            if full_text.strip():
                # Split into logical blocks while preserving order
                paragraphs = [p.strip() for p in full_text.split('\n\n') if p.strip()]
                log.info(f"Extracted {len(paragraphs)} text blocks from document")
                
                for idx, paragraph in enumerate(paragraphs):
                    text_element = {
                        "element_id": f"text_{idx + 1}",
                        "element_type": "text",
                        "content": clean_text(paragraph),
                        "order": idx + 1,
                        "page_number": 1,  # Docling doesn't always provide accurate page numbers
                        "metadata": {
                            "extraction_method": "docling_comprehensive",
                            "extraction_type": "combined_text_ocr",
                            "block_index": idx,
                            "total_blocks": len(paragraphs),
                            "ocr_used": "auto_detected"
                        },
                        "timestamp": datetime.datetime.now().isoformat()
                    }
                    text_elements.append(text_element)
        
        # Fallback to individual text elements if available
        if not text_elements and hasattr(doc, 'texts') and doc.texts:
            log.info(f"Using individual text elements: {len(doc.texts)} found")
            for idx, text_item in enumerate(doc.texts):
                if hasattr(text_item, 'content') and str(text_item.content).strip():
                    text_element = {
                        "element_id": f"text_{idx + 1}",
                        "element_type": "text",
                        "content": clean_text(str(text_item.content).strip()),
                        "order": idx + 1,
                        "page_number": 1,
                        "metadata": {
                            "extraction_method": "docling_individual_elements",
                            "extraction_type": "granular_text",
                            "block_index": idx,
                            "total_blocks": len(doc.texts)
                        },
                        "timestamp": datetime.datetime.now().isoformat()
                    }
                    text_elements.append(text_element)
        
        log.info(f"Docling extraction complete: {len(text_elements)} text elements extracted")
        return text_elements, page_cnt
        
    except Exception as e:
        log.error(f"Docling extraction failed: {e}")
        raise

# ---------------------------------------------------------------------------

def extract_pdf_to_json(
    pdf_path: str | Path,
    output_dir: str | Path | None = None,
    config_overrides: Optional[Dict[str, Any]] = None
) -> Path:
    """
    Convert *pdf_path* → JSON and return JSON path.

    Args:
        pdf_path: Path to PDF file
        output_dir: Directory for output JSON (defaults to same as PDF)
        config_overrides: Optional dict of config overrides
        
    Returns:
        Path to generated JSON file
        
    JSON structure matches earlier code (text + optional images with OCR).
    """
    # Get configuration with any overrides
    config = get_config(config_overrides)
    
    p = Path(pdf_path).resolve()
    if not p.exists() or p.suffix.lower() != ".pdf":
        raise FileNotFoundError(f"PDF not found: {p}")

    out_dir = Path(output_dir or p.parent).resolve()
    out_dir.mkdir(exist_ok=True)

    text_elems, page_cnt = _docling_extract(p)
    word_cnt = sum(len(e["content"].split()) for e in text_elems)

    img_elems: List[Dict[str, Any]] = []
    if word_cnt < config.MIN_WORDS:
        log.info("Few words (%d) – extracting images with OCR=%s …",
                 word_cnt, config.ENABLE_IMAGE_OCR)
        img_elems = _extract_images(p, config)
        # order = after all text
        for i, e in enumerate(img_elems, 1):
            e["order"] = len(text_elems) + i

    all_elems = text_elems + img_elems
    out_json = {
        "file_metadata": {
            "filename": p.name,
            "total_pages": page_cnt,
            "word_count": word_cnt,
            "extracted_at": datetime.datetime.now().isoformat(),
            "ocr_enabled": config.ENABLE_IMAGE_OCR,
            "ocr_engine": config.OCR_ENGINE if config.ENABLE_IMAGE_OCR else None,
        },
        "elements": all_elems,
    }

    json_path = out_dir / f"{p.stem}_dual.json"
    json_path.write_text(json.dumps(out_json, indent=2, ensure_ascii=False))
    log.info("✅ Extracted → %s", json_path)
    return json_path

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Extract PDF → JSON (text + optional images)"
    )
    ap.add_argument("pdf", type=Path)
    ap.add_argument("-o", "--output-dir", type=Path, help="Where to write JSON")
    args = ap.parse_args()

    extract_pdf_to_json(args.pdf, args.output_dir)