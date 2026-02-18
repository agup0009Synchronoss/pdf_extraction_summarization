"""
DOTS Extractor Module
=====================

Real DOTS OCR extraction via HuggingFace Transformers (in-process):
- Rasterize PDF pages with PyMuPDF (fitz)
- Resize images to model-friendly pixel bounds
- Run rednote-hilab/dots.ocr via model.generate() locally
- Parse response and normalize for prompt bridge
"""

import json
import logging
import math
import tempfile
import torch
from pathlib import Path
from typing import Dict, Any, List, Optional
import datetime

log = logging.getLogger("dots_extractor")

# DOTS layout prompt (prompt_layout_all_en)
DOTS_LAYOUT_PROMPT = """Please output the layout information from the PDF image, including each layout element's bbox, its category, and the corresponding text content within the bbox.

1. Bbox format: [x1, y1, x2, y2]

2. Layout Categories: The possible categories are ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title'].

3. Text Extraction & Formatting Rules:
 - Picture: For the 'Picture' category, the text field should be omitted.
 - Formula: Format its text as LaTeX.
 - Table: Format its text as HTML.
 - All Others (Text, Title, etc.): Format their text as Markdown.

4. Constraints:
 - The output text must be the original text from the image, with no translation.
 - All layout elements must be sorted according to human reading order.

5. Final Output: The entire output must be a single JSON object.
"""

# Pixel bounds for model input (from DOTS image_utils / consts)
IMAGE_FACTOR = 28
MIN_PIXELS = 3136
MAX_PIXELS = 11_289_600

# Map DOTS category to internal element type
DOTS_CATEGORY_TO_TYPE = {
    "Title": "text",
    "Text": "text",
    "Section-header": "text",
    "Page-header": "text",
    "Page-footer": "text",
    "Caption": "text",
    "Footnote": "text",
    "List-item": "text",
    "Table": "table",
    "Formula": "formula",
    "Picture": "picture",
}


def _round_by_factor(number: int, factor: int) -> int:
    return round(number / factor) * factor


def _floor_by_factor(number: int, factor: int) -> int:
    return math.floor(number / factor) * factor


def _ceil_by_factor(number: int, factor: int) -> int:
    return math.ceil(number / factor) * factor


def smart_resize(
    height: int,
    width: int,
    factor: int = IMAGE_FACTOR,
    min_pixels: int = MIN_PIXELS,
    max_pixels: int = MAX_PIXELS,
) -> tuple:
    """Rescale dimensions so total pixels in [min_pixels, max_pixels] and divisible by factor."""
    if max(height, width) / max(min(height, width), 1) > 200:
        raise ValueError("Aspect ratio must be smaller than 200")
    h_bar = max(factor, _round_by_factor(height, factor))
    w_bar = max(factor, _round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, _floor_by_factor(height / beta, factor))
        w_bar = max(factor, _floor_by_factor(width / beta, factor))
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = _ceil_by_factor(int(height * beta), factor)
        w_bar = _ceil_by_factor(int(width * beta), factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((h_bar * w_bar) / max_pixels)
        h_bar = max(factor, _floor_by_factor(h_bar / beta, factor))
        w_bar = max(factor, _floor_by_factor(w_bar / beta, factor))
    return h_bar, w_bar


class DotsExtractor:
    """
    DOTS extraction service using HuggingFace Transformers (in-process inference).

    Interface contract:
    - Input: PDF path, page options, quality options
    - Output: Raw extraction result + normalized result
    """

    def __init__(self, config):
        self.config = config
        self.effective_device = config.get_effective_device()
        self._model = None
        self._processor = None
        log.info(
            "DotsExtractor initialized (model=%s, device=%s)",
            config.DOTS_MODEL_PATH,
            self.effective_device,
        )

    def _resolve_model_path(self) -> str:
        """
        Resolve the model path to use for loading.

        transformers>=4.49 derives a Python module name from the HF repo ID when
        trust_remote_code=True.  A dot in the repo name (e.g. 'dots.ocr') makes
        Python treat it as a sub-package import and raises:
            ModuleNotFoundError: No module named 'transformers_modules.rednote-hilab.dots'

        The official dots.ocr docs note: "use a directory name without periods".
        Fix: if the HF model ID contains a dot, snapshot-download it once to a
        safe local directory (dots replaced with underscores) and load from there.
        """
        model_path = self.config.DOTS_MODEL_PATH

        # Already a local path — use as-is.
        if Path(model_path).exists():
            return model_path

        # HF repo ID with a dot in the model name → needs local download.
        if "/" in model_path:
            model_name = model_path.split("/", 1)[1]
            if "." in model_name:
                safe_name = model_name.replace(".", "_")
                local_dir = Path("./hf_cache") / safe_name
                if (local_dir / "config.json").exists():
                    log.info("Using cached local model at: %s", local_dir)
                    return str(local_dir)
                log.info(
                    "Downloading %s to local path %s (avoids dots-in-module-name error)...",
                    model_path, local_dir,
                )
                from huggingface_hub import snapshot_download
                snapshot_download(model_path, local_dir=str(local_dir))
                return str(local_dir)

        return model_path

    def _load_model(self):
        """Lazy-load model and processor on first call to avoid GPU memory at import time."""
        if self._model is not None:
            return
        from transformers import AutoModelForCausalLM, AutoProcessor

        resolved_path = self._resolve_model_path()
        log.info("Loading DOTS model from %s ...", resolved_path)
        self._model = AutoModelForCausalLM.from_pretrained(
            resolved_path,
            attn_implementation=self.config.DOTS_ATTN_IMPLEMENTATION,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        self._processor = AutoProcessor.from_pretrained(
            resolved_path,
            trust_remote_code=True,
            use_fast=True,
        )
        log.info("DOTS model loaded successfully (device: %s)", self.effective_device)

    def extract_pdf(self, pdf_path: Path, page_cap: Optional[int] = None) -> Dict[str, Any]:
        page_cap = page_cap or self.config.PAGE_CAP
        dpi = self.config.DPI

        log.info("Extracting PDF: %s (page_cap=%s, dpi=%s)", pdf_path.name, page_cap, dpi)

        raw_result, debug_dir = self._run_extraction(pdf_path, page_cap, dpi)
        normalized_result = self._normalize_extraction(raw_result, pdf_path)

        metadata = {
            "pdf_name": pdf_path.name,
            "page_cap": page_cap,
            "dpi": dpi,
            "device_policy": self.config.DEVICE_POLICY,
            "effective_device": self.effective_device,
            "backend": "transformers",
            "model_path": self.config.DOTS_MODEL_PATH,
            "extraction_timestamp": datetime.datetime.now().isoformat(),
        }
        if debug_dir is not None:
            metadata["debug_dir"] = str(debug_dir)

        return {
            "raw": raw_result,
            "normalized": normalized_result,
            "metadata": metadata,
        }

    def _run_extraction(self, pdf_path: Path, page_cap: int, dpi: int) -> tuple:
        """Rasterize PDF, run model per page, aggregate raw results. Returns (raw_result, debug_dir or None)."""
        import fitz
        from PIL import Image

        debug_dir = None
        if getattr(self.config, "DEBUG_SAVE_ARTIFACTS", False):
            base = Path(getattr(self.config, "DEBUG_OUTPUT_DIR", "debug_output"))
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            debug_dir = base / f"{pdf_path.stem}_{ts}"
            debug_dir.mkdir(parents=True, exist_ok=True)
            log.info("Debug artifacts will be saved to: %s", debug_dir)

        # Load model once before processing pages
        self._load_model()

        pages_out = []
        with fitz.open(pdf_path) as doc:
            total_pages = doc.page_count
            end_page = min(page_cap, total_pages)

            for page_idx in range(end_page):
                page_num = page_idx + 1
                page = doc[page_idx]
                pil_image = self._rasterize_page(page, dpi)
                if pil_image is None:
                    log.warning("Page %s rasterization failed, skipping", page_num)
                    pages_out.append({"page_num": page_num, "elements": [], "reading_order": []})
                    continue
                pil_resized = self._resize_page_image(pil_image)
                if debug_dir is not None:
                    img_path = debug_dir / f"page_{page_num}_image.png"
                    pil_resized.save(img_path)
                elements, raw_response = self._call_transformers(pil_resized, page_num, debug_dir)
                if debug_dir is not None and raw_response is not None:
                    (debug_dir / f"page_{page_num}_model_response.txt").write_text(raw_response, encoding="utf-8")
                    parsed_path = debug_dir / f"page_{page_num}_model_parsed.json"
                    parsed_path.write_text(json.dumps(elements, indent=2, ensure_ascii=False), encoding="utf-8")
                reading_order = list(range(len(elements)))
                pages_out.append({
                    "page_num": page_num,
                    "width": pil_resized.width,
                    "height": pil_resized.height,
                    "elements": elements,
                    "reading_order": reading_order,
                })

        raw_result = {
            "document": pdf_path.name,
            "total_pages": total_pages,
            "processed_pages": len(pages_out),
            "dpi": dpi,
            "pages": pages_out,
            "format_version": "dots_transformers_1.0",
        }
        return raw_result, debug_dir

    def _rasterize_page(self, page, dpi: int):
        """Render a single fitz page to PIL Image."""
        import fitz
        from PIL import Image

        try:
            mat = fitz.Matrix(dpi / 72, dpi / 72)
            pm = page.get_pixmap(matrix=mat, alpha=False)
            if pm.width > 4500 or pm.height > 4500:
                mat = fitz.Matrix(72 / 72, 72 / 72)
                pm = page.get_pixmap(matrix=mat, alpha=False)
            return Image.frombytes("RGB", (pm.width, pm.height), pm.samples)
        except Exception as e:
            log.warning("Rasterize failed: %s", e)
            return None

    def _resize_page_image(self, pil_image):
        """Resize image to model-friendly pixel bounds."""
        from PIL import Image

        w, h = pil_image.size
        new_h, new_w = smart_resize(h, w)
        if (new_h, new_w) == (h, w):
            return pil_image
        return pil_image.resize((new_w, new_h), Image.Resampling.LANCZOS)

    def _call_transformers(self, pil_image, page_num: int, debug_dir: Optional[Path] = None) -> tuple:
        """Run DOTS model on one page image and return (parsed elements, raw response text or None)."""
        from qwen_vl_utils import process_vision_info

        try:
            # Save PIL image to a temp file so the processor can handle it via standard path
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp_path = tmp.name
            pil_image.save(tmp_path)

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": tmp_path},
                        {"type": "text", "text": DOTS_LAYOUT_PROMPT},
                    ],
                }
            ]

            text = self._processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self._processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.effective_device)

            with torch.no_grad():
                generated_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=self.config.DOTS_MAX_NEW_TOKENS,
                )

            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_texts = self._processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            content = output_texts[0] if output_texts else ""

            # Clean up temp file
            try:
                Path(tmp_path).unlink()
            except Exception:
                pass

            if not content:
                log.warning("Empty model response for page %s", page_num)
                return [], None

            elements = self._parse_model_response(content, page_num)
            return elements, content

        except Exception as e:
            log.warning("Model inference failed for page %s: %s", page_num, e)
            return [], None

    def _parse_model_response(self, raw_text: str, page_num: int) -> List[Dict[str, Any]]:
        """Lenient parse of DOTS JSON response into elements with type/bbox/text/html/latex."""
        text = raw_text.strip()
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            if "```json" in text:
                start = text.find("```json") + 7
                end = text.find("```", start)
                if end > start:
                    try:
                        data = json.loads(text[start:end].strip())
                    except json.JSONDecodeError:
                        log.warning("Page %s: could not parse JSON from markdown block", page_num)
                        return []
                else:
                    log.warning("Page %s: malformed markdown JSON block", page_num)
                    return []
            else:
                log.warning("Page %s: response is not valid JSON", page_num)
                return []

        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            items = data.get("elements") or data.get("cells")
            if items is None:
                for v in data.values():
                    if isinstance(v, list):
                        items = v
                        break
            if not isinstance(items, list):
                items = []
        else:
            items = []

        out = []
        for el in items:
            if not isinstance(el, dict):
                continue
            bbox = el.get("bbox", [])
            category = el.get("category", "Text")
            el_text = el.get("text", "")
            elem_type = DOTS_CATEGORY_TO_TYPE.get(category, "text")
            if elem_type == "picture":
                continue  # no text field per DOTS spec
            rec = {
                "type": elem_type,
                "bbox": bbox,
                "category": category,
                "confidence": el.get("confidence", 0.0),
            }
            if elem_type == "table":
                rec["html"] = el_text
            elif elem_type == "formula":
                rec["latex"] = el_text
            else:
                rec["text"] = el_text
            out.append(rec)
        return out

    def _normalize_extraction(self, raw_result: Dict[str, Any], pdf_path: Path) -> Dict[str, Any]:
        """Normalize raw DOTS output into stable schema for prompt builder. Lenient."""
        warnings = []
        pages_data = raw_result.get("pages", [])
        if not pages_data:
            warnings.append("No pages found in raw extraction")

        all_elements = []
        element_id = 1
        for page in pages_data:
            page_num = page.get("page_num", 0)
            elements = page.get("elements", [])
            reading_order = page.get("reading_order", list(range(len(elements))))

            for order_idx, elem_idx in enumerate(reading_order):
                if elem_idx >= len(elements):
                    warnings.append(f"Invalid reading order index {elem_idx} on page {page_num}")
                    continue
                elem = elements[elem_idx]
                elem_type = elem.get("type", "unknown")
                normalized_elem = {
                    "element_id": f"elem_{element_id}",
                    "element_type": elem_type,
                    "page_number": page_num,
                    "order": order_idx + 1,
                    "bbox": elem.get("bbox", []),
                    "confidence": elem.get("confidence", 0.0),
                    "category": elem.get("category", "unknown"),
                }
                if elem_type == "text":
                    normalized_elem["content"] = elem.get("text", "")
                elif elem_type == "table":
                    normalized_elem["content"] = elem.get("html", "")
                    normalized_elem["format"] = "html"
                elif elem_type == "formula":
                    normalized_elem["content"] = elem.get("latex", "")
                    normalized_elem["format"] = "latex"
                else:
                    normalized_elem["content"] = str(elem.get("text", ""))
                    if elem_type not in ("text", "table", "formula"):
                        warnings.append(f"Unknown element type: {elem_type}")
                all_elements.append(normalized_elem)
                element_id += 1

        normalized = {
            "file_metadata": {
                "filename": pdf_path.name,
                "total_pages": raw_result.get("total_pages", 0),
                "processed_pages": raw_result.get("processed_pages", 0),
                "extracted_at": datetime.datetime.now().isoformat(),
                "extraction_method": "dots_ocr",
                "format_version": raw_result.get("format_version", "unknown"),
            },
            "elements": all_elements,
            "warnings": warnings,
        }
        if warnings:
            log.warning("Normalization warnings: %s", warnings)
        return normalized


def extract_pdf_with_dots(pdf_path: Path, config) -> Dict[str, Any]:
    """Convenience: extract PDF with DOTS (HF Transformers backend)."""
    extractor = DotsExtractor(config)
    return extractor.extract_pdf(pdf_path)
