"""
DOTS OCR Pipeline Orchestrator
==============================

Ties together: PDF -> DOTS extraction -> normalization -> prompt -> Ollama -> results
"""

import logging
from pathlib import Path
from typing import Dict, Any
import json
import datetime

from config import get_config, DotsConfig
from extractor import extract_pdf_with_dots
from prompt_bridge import build_ollama_prompt
from ollama_client import call_ollama
from call_llama_api import API_URL, MODEL  # for logging only

log = logging.getLogger("dots_pipeline")


class DotsPipeline:
    """
    End-to-end pipeline orchestrator for DOTS OCR workflow.
    """
    
    def __init__(self, config: DotsConfig):
        """
        Initialize pipeline with configuration.
        
        Args:
            config: DotsConfig instance
        """
        self.config = config
        self.output_dir = Path(config.OUTPUT_DIR)
        self.output_dir.mkdir(exist_ok=True)
        
        log.info("DotsPipeline initialized (LLM: %s, model: %s)", API_URL, MODEL)
    
    def process_pdf(self, pdf_path: Path, page_cap_override: int = None) -> Dict[str, Any]:
        """
        Process a single PDF through the full DOTS pipeline.
        
        Args:
            pdf_path: Path to PDF file
            page_cap_override: Optional override for page cap
            
        Returns:
            Complete processing result with all artifacts and metadata
        """
        log.info(f"Processing PDF: {pdf_path}")
        start_time = datetime.datetime.now()
        
        try:
            # Stage 1: DOTS Extraction
            log.info("Stage 1: DOTS extraction")
            extraction_result = extract_pdf_with_dots(pdf_path, self.config)
            
            # Stage 2: Persist raw and normalized JSON
            log.info("Stage 2: Persisting extraction artifacts")
            raw_json_path = self._save_raw_json(pdf_path, extraction_result["raw"])
            normalized_json_path = self._save_normalized_json(pdf_path, extraction_result["normalized"])
            
            # Stage 3: Build Ollama prompt
            log.info("Stage 3: Building Ollama prompt")
            prompt = build_ollama_prompt(extraction_result["normalized"], self.config)
            prompt_path = self._save_prompt(pdf_path, prompt)

            # Debug: write prompt and extraction JSONs into timestamped debug folder
            debug_dir = extraction_result.get("metadata", {}).get("debug_dir")
            if debug_dir:
                self._write_debug_artifacts(
                    Path(debug_dir),
                    prompt=prompt,
                    raw_extraction=extraction_result["raw"],
                    normalized_extraction=extraction_result["normalized"],
                )
            
            # Stage 4: Call Ollama for summary/classification
            log.info("Stage 4: Calling Ollama API")
            ollama_response = call_ollama(prompt, self.config)
            
            # Stage 5: Persist final response
            log.info("Stage 5: Persisting Ollama response")
            response_path = self._save_response(pdf_path, prompt, ollama_response, extraction_result["metadata"])
            
            end_time = datetime.datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # Build final result
            result = {
                "status": "success",
                "pdf_name": pdf_path.name,
                "summary": ollama_response["summary"],
                "classification": ollama_response["classification_label"],
                "artifacts": {
                    "raw_extraction": str(raw_json_path),
                    "normalized_extraction": str(normalized_json_path),
                    "prompt": str(prompt_path),
                    "response": str(response_path)
                },
                "metadata": {
                    "processing_time_seconds": duration,
                    "extraction_method": "dots_ocr",
                    "device_used": extraction_result["metadata"]["effective_device"],
                    "page_cap": extraction_result["metadata"]["page_cap"],
                    "dpi": extraction_result["metadata"]["dpi"],
                    "warnings": extraction_result["normalized"].get("warnings", []),
                    "timestamp": datetime.datetime.now().isoformat(),
                }
            }
            if debug_dir:
                result["artifacts"]["debug_dir"] = debug_dir
                result["metadata"]["debug_dir"] = debug_dir
            
            log.info(f"Pipeline complete: {pdf_path.name} ({duration:.2f}s)")
            log.info(f"  Classification: {result['classification']}")
            
            return result
            
        except Exception as e:
            log.error(f"Pipeline failed for {pdf_path.name}: {str(e)}")
            return {
                "status": "error",
                "pdf_name": pdf_path.name,
                "error": str(e),
                "timestamp": datetime.datetime.now().isoformat()
            }
    
    def _save_raw_json(self, pdf_path: Path, raw_extraction: Dict[str, Any]) -> Path:
        """Save raw DOTS extraction JSON."""
        output_path = self.output_dir / f"{pdf_path.stem}_raw_dots.json"
        output_path.write_text(json.dumps(raw_extraction, indent=2, ensure_ascii=False), encoding="utf-8")
        return output_path
    
    def _save_normalized_json(self, pdf_path: Path, normalized_extraction: Dict[str, Any]) -> Path:
        """Save normalized extraction JSON."""
        output_path = self.output_dir / f"{pdf_path.stem}_normalized.json"
        output_path.write_text(json.dumps(normalized_extraction, indent=2, ensure_ascii=False), encoding="utf-8")
        return output_path
    
    def _save_prompt(self, pdf_path: Path, prompt: str) -> Path:
        """Save generated prompt text."""
        output_path = self.output_dir / f"{pdf_path.stem}_prompt.txt"
        output_path.write_text(prompt, encoding="utf-8")
        return output_path
    
    def _write_debug_artifacts(
        self,
        debug_dir: Path,
        prompt: str,
        raw_extraction: Dict[str, Any],
        normalized_extraction: Dict[str, Any],
    ) -> None:
        """Write prompt and extraction JSONs into the timestamped debug folder."""
        debug_dir = Path(debug_dir)
        if not debug_dir.exists():
            return
        (debug_dir / "prompt_sent_to_llm.txt").write_text(prompt, encoding="utf-8")
        (debug_dir / "extraction_raw.json").write_text(
            json.dumps(raw_extraction, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (debug_dir / "extraction_normalized.json").write_text(
            json.dumps(normalized_extraction, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        log.info("Debug artifacts written to %s", debug_dir)

    def _save_response(self, pdf_path: Path, prompt: str, ollama_response: Dict[str, Any], extraction_metadata: Dict[str, Any]) -> Path:
        """Save complete prompt + response artifact."""
        output_data = {
            "pdf_file": pdf_path.name,
            "prompt": prompt,
            "ollama_response": {
                "summary": ollama_response["summary"],
                "classification_label": ollama_response["classification_label"],
                "model": ollama_response["model"],
                "timestamp": ollama_response["timestamp"]
            },
            "extraction_metadata": extraction_metadata
        }
        
        output_path = self.output_dir / f"{pdf_path.stem}_result.json"
        output_path.write_text(json.dumps(output_data, indent=2, ensure_ascii=False), encoding="utf-8")
        return output_path


def run_pipeline(pdf_path: Path, config_overrides: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Convenience function to run the pipeline.
    
    Args:
        pdf_path: Path to PDF file
        config_overrides: Optional configuration overrides
        
    Returns:
        Processing result
    """
    config = get_config(config_overrides)
    pipeline = DotsPipeline(config)
    return pipeline.process_pdf(pdf_path)
