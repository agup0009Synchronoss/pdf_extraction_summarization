"""
PDF EXTRACTOR
=============

Single-file implementation of:
 • Docling text extraction  (embedded-text + OCR)
 • Optional pikepdf image extraction + BLIP captioning
 • Outputs one <stem>_dual.json next to the source or to --output-dir
"""

from __future__ import annotations
import base64, datetime, io, json, logging, sys, tempfile, time
from pathlib import Path
from typing import Any, Dict, List

import pikepdf
from pikepdf import PdfImage
from PIL import Image

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

# --- tunables --------------------------------------------------------------
MIN_WORDS = 200     # if text < threshold → also extract images
MAX_IMAGES = 5
MIN_AREA  = 50_000  # skip images smaller than this (w*h)

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

def _extract_images(pdf_path: Path) -> List[Dict[str, Any]]:
    """Return image elements with captions (largest by pixel area)."""
    pdf = pikepdf.open(pdf_path)
    meta: List[Dict[str, Any]] = []

    for p_idx, page in enumerate(pdf.pages, 1):
        for name, obj in page.images.items():
            img = PdfImage(obj)
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                out = img.extract_to(fileprefix=tmp.name.rstrip(".png"))
            try:
                pil = Image.open(out).convert("RGB")
            finally:
                Path(out).unlink(missing_ok=True)

            w, h = pil.size
            area = w * h
            if area < MIN_AREA:
                continue
            meta.append({"pil": pil, "page": p_idx, "name": name, "area": area})

    if not meta:
        return []

    meta.sort(key=lambda m: m["area"], reverse=True)
    selected = meta[:MAX_IMAGES]
    elements: List[Dict[str, Any]] = []

    for idx, m in enumerate(selected, 1):
        pil = m["pil"]
        pil224 = pil.resize((224, 224))
        buffer = io.BytesIO()
        pil224.save(buffer, format="JPEG", quality=85)
        elements.append(
            {
                "element_id": f"image_{idx}",
                "element_type": "image",
                "page_number": m["page"],
                "order": 0,  # will be set by caller
                "caption": _caption(pil224),
                "image_224_jpeg_base64": base64.b64encode(buffer.getvalue())
                .decode(),
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

def extract_pdf_to_json(pdf_path: str | Path, output_dir: str | Path | None = None) -> Path:
    """
    Convert *pdf_path* → JSON and return JSON path.

    JSON structure matches earlier code (text + optional images).
    """
    p = Path(pdf_path).resolve()
    if not p.exists() or p.suffix.lower() != ".pdf":
        raise FileNotFoundError(f"PDF not found: {p}")

    out_dir = Path(output_dir or p.parent).resolve()
    out_dir.mkdir(exist_ok=True)

    text_elems, page_cnt = _docling_extract(p)
    word_cnt = sum(len(e["content"].split()) for e in text_elems)

    img_elems: List[Dict[str, Any]] = []
    if word_cnt < MIN_WORDS:
        log.info("Few words (%d) – extracting images …", word_cnt)
        img_elems = _extract_images(p)
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