"""
pdf_pipeline.py  –  MAIN ORCHESTRATOR
=====================================

Usage
  python pdf_pipeline.py <pdf_file | directory> <output_dir> [--overwrite]

Steps
  1.  pdf_extractor.extract_pdf_to_json  → writes <stem>_dual.json + returns path
  2.  builds prompt   (in-memory)
  3.  call_llama_api.send_prompt        → gets response
  4.  writes
        • JSON                         (already done in step-1)
        • <stem>_prompt_response.json  (structured prompt + LLM response)
        • llama_responses.json         (aggregate log)
"""

from __future__ import annotations
import datetime, json, logging, textwrap
from pathlib import Path
from typing import Dict, Any, List, Optional
import requests

from pdf_extractor import extract_pdf_to_json
from call_llama_api import send_prompt, API_KEY
from config import get_config

# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger("pdf_pipeline")

# ---------------------------------------------------------------------------
# # Text cleaning function to handle OCR artifacts and Unicode issues
# def clean_text(text: str) -> str:
#     """Clean OCR text by removing problematic Unicode characters and artifacts."""
#     if not text:
#         return text
    
#     # Remove or replace problematic Unicode characters
#     import re
    
#     # Replace common problematic Unicode characters with ASCII equivalents
#     replacements = {
#         '\u01af': 'U',  # Latin capital letter U with horn
#         '\u01b0': 'u',  # Latin small letter u with horn
#         '\u2217': '*',  # Asterisk operator
#         '\u2013': '-',  # En dash
#         '\u2014': '--', # Em dash
#         '\u2018': "'",  # Left single quotation mark
#         '\u2019': "'",  # Right single quotation mark
#         '\u201c': '"',  # Left double quotation mark
#         '\u201d': '"',  # Right double quotation mark
#         '\u2022': '•',  # Bullet
#         '\u00a0': ' ',  # Non-breaking space
#         '\u00b0': '°',  # Degree sign
#         '\u00ae': '®',  # Registered trademark
#         '\u00a9': '©',  # Copyright
#         '\u2122': '™',  # Trademark
#     }
    
#     cleaned = text
#     for unicode_char, replacement in replacements.items():
#         cleaned = cleaned.replace(unicode_char, replacement)
    
#     # Remove other non-printable characters except basic whitespace
#     cleaned = re.sub(r'[^\x20-\x7E\n\r\t]', '', cleaned)
    
#     # Normalize whitespace
#     cleaned = re.sub(r'\s+', ' ', cleaned)
#     cleaned = cleaned.strip()
    
#     return cleaned

# ---------------------------------------------------------------------------
# prompt builder (kept here for full in-memory flow)
def create_prompt(obj: Dict[str, Any], config_overrides: Optional[Dict[str, Any]] = None) -> str:
    """
    Build LLM prompt from extracted document elements.
    
    Args:
        obj: Extraction JSON object with elements
        config_overrides: Optional config overrides (for MAX_PROMPT_WORDS)
        
    Returns:
        Formatted prompt string
    """
    config = get_config(config_overrides)
    elems = sorted(obj["elements"], key=lambda e: e["order"])
    
    # Build the base prompt
    prompt = textwrap.dedent(
            """\
            You are an expert document analyst.

            TASKS
            1. Summarize the document in 2–3 crisp sentences.
            2. Classify the document into exactly ONE category from the list below:

            advertisement, budget, email, ID, form, handwritten, invoice,
            letter, memo, news_article, presentation, questionnaire,
            resume, scientific_document, specification, tax_record,
            bank_card, financial_statement, lease_agreement, property_deed,
            insurance_document, vehicle_title, unclassified

            Return ONLY:
            {
            "summary": "...",
            "classification_label": "..."
            }

            DOCUMENT Chunks:
            """

    )
    
    # Collect all content first to count words
    content_parts = []
    total_words = 0
    max_words = config.MAX_PROMPT_WORDS
    
    for idx, e in enumerate(elems, 1):
        if e["element_type"] == "text":
            content = f"\nChunk {idx} [text]: {e['content']}"
        else:
            # Combine BLIP caption + OCR text for images
            parts = [f"\nChunk {idx} [image]:"]
            
            # Add caption if available and valid
            caption = e.get('caption', '')
            if caption and caption not in ('', '<BLIP unavailable>', '<caption error>'):
                parts.append(f"Visual Description: {caption}")
            
            # Add OCR text if available
            ocr_text = e.get('ocr_text', '')
            if ocr_text:
                parts.append(f"Extracted Text: {ocr_text}")
            
            # Join parts or add fallback
            if len(parts) > 1:
                content = ' | '.join(parts)
            else:
                content = parts[0] + " (no caption/OCR)"
        
        # Count words in this chunk
        chunk_words = len(content.split())
        
        # Check if adding this chunk would exceed the limit
        if total_words + chunk_words > max_words:
            # Add partial content if we haven't added anything yet
            if total_words == 0:
                # Take first part of the chunk to reach max_words
                words_needed = max_words - total_words
                words_in_chunk = content.split()[:words_needed]
                content = ' '.join(words_in_chunk) + "..."
                content_parts.append(content)
                log.info(f"Content truncated to {max_words} words (limit reached)")
            break
        
        content_parts.append(content)
        total_words += chunk_words
    
    # Add all collected content to prompt
    prompt += ''.join(content_parts)
    
    log.info(f"Prompt created with {total_words} words of content")
    return prompt

# ---------------------------------------------------------------------------
def write_prompt_response(out_dir: Path, stem: str,
                          prompt: str, response: requests.Response) -> Path:
    out = out_dir / f"{stem}_prompt_response.json"
    
    # Parse the response JSON to extract the content
    try:
        response_json = response.json()
        response_content = response_json.get("message", {}).get("content", "")
        
        # Try to parse the content as JSON (the actual LLM response)
        try:
            parsed_content = json.loads(response_content)
        except json.JSONDecodeError:
            # If content is not valid JSON, keep it as string
            parsed_content = response_content
        
        # Create the structured output
        output_data = {
            "llm_prompt": {
                "content": prompt
            },
            "response_body": {
                "model": response_json.get("model", ""),
                "created_at": response_json.get("created_at", ""),
                "total_duration": response_json.get("total_duration", 0),
                "load_duration": response_json.get("load_duration", 0),
                "prompt_eval_count": response_json.get("prompt_eval_count", 0),
                "prompt_eval_duration": response_json.get("prompt_eval_duration", 0),
                "eval_count": response_json.get("eval_count", 0),
                "eval_duration": response_json.get("eval_duration", 0),
                "done_reason": response_json.get("done_reason", ""),
                "done": response_json.get("done", False),
                "content": parsed_content
            }
        }
        
        out.write_text(json.dumps(output_data, indent=2, ensure_ascii=False), encoding="utf-8")
        return out
        
    except Exception as e:
        # Fallback to simple structure if parsing fails
        output_data = {
            "llm_prompt": {
                "content": prompt
            },
            "response_body": {
                "raw_response": response.text,
                "status_code": response.status_code,
                "error": f"Failed to parse response: {str(e)}"
            }
        }
        out.write_text(json.dumps(output_data, indent=2, ensure_ascii=False), encoding="utf-8")
        return out

def process_pdf(
    pdf: Path,
    out_dir: Path,
    agg: List[Dict[str, Any]],
    config_overrides: Optional[Dict[str, Any]] = None
) -> None:
    """Process a single PDF through the full pipeline."""
    json_path = extract_pdf_to_json(pdf, out_dir, config_overrides)
    json_obj  = json.loads(json_path.read_text(encoding="utf-8"))

    prompt = create_prompt(json_obj, config_overrides)
    resp   = send_prompt(prompt, API_KEY)

    combined_json = write_prompt_response(out_dir, pdf.stem, prompt, resp)

    agg.append({
        "pdf_file": pdf.name,
        "json": json_path.name,
        "combined": combined_json.name,
        "status_code": resp.status_code,
        "timestamp": datetime.datetime.now().isoformat(),
    })
    log.info("Done %s  (status %s)", pdf.name, resp.status_code)

# ---------------------------------------------------------------------------
def discover_pdfs(p: Path) -> List[Path]:
    return [p] if p.is_file() else sorted(p.rglob("*.pdf"))

def run(
    input_path: Path,
    output_dir: Path,
    overwrite: bool,
    config_overrides: Optional[Dict[str, Any]] = None
) -> None:
    """
    Run the full pipeline on PDFs.
    
    Args:
        input_path: PDF file or directory
        output_dir: Output directory
        overwrite: Whether to overwrite existing aggregate JSON
        config_overrides: Optional config overrides for this run
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    agg_path = output_dir / "llama_responses.json"
    aggregated: List[Dict[str, Any]] = [] if overwrite or not agg_path.exists() \
                                      else json.loads(agg_path.read_text())

    pdfs = discover_pdfs(input_path)
    if not pdfs:
        log.error("No PDFs found at %s", input_path)
        return

    for pdf in pdfs:
        try:
            process_pdf(pdf, output_dir, aggregated, config_overrides)
        except Exception as e:
            log.error("❌ %s failed: %s", pdf.name, e)
            aggregated.append({
                "pdf_file": pdf.name,
                "status": "error",
                "error": str(e),
                "timestamp": datetime.datetime.now().isoformat(),
            })

    agg_path.write_text(json.dumps(aggregated, indent=2, ensure_ascii=False))
    log.info("✅ Pipeline finished. Aggregate → %s", agg_path)

# --------------------------------------------------------------------------- CLI
if __name__ == "__main__":
    import argparse, sys

    ap = argparse.ArgumentParser(
        description="End-to-end PDF → JSON → Prompt → Ollama pipeline"
    )
    ap.add_argument("input",  type=Path, help="PDF file or directory")
    ap.add_argument("output", type=Path, help="Directory for outputs")
    ap.add_argument("--overwrite", action="store_true",
                    help="Replace existing llama_responses.json")
    args = ap.parse_args()

    run(args.input.resolve(), args.output.resolve(), args.overwrite)