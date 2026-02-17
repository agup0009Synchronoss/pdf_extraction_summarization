"""
Ollama Client Module
====================

Handles LLM API calls for summary and classification using the same remote
cron-job service as the main pdf_extraction app (call_llama_api.py).

Fail-fast: raises exceptions on any failure, no fallback path.
"""

from __future__ import annotations

import json
import logging
import datetime
from typing import Dict, Any

# Reuse the same API wrapper that the main pdf_extraction app uses.
# This gives us identical endpoint, auth header (X-Api-Key), model, system
# message, temperature/seed options, and response shape.
from call_llama_api import send_prompt as _send_prompt, API_KEY, API_URL, MODEL

log = logging.getLogger("ollama_client")


# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------

def _parse_llm_response(response, classification_labels: list[str]) -> Dict[str, Any]:
    """
    Parse the raw requests.Response from the LLM API.

    The API returns:
        { "message": { "content": "<JSON string>" }, ... }

    Args:
        response: requests.Response from call_llama_api.send_prompt
        classification_labels: Valid label list for validation

    Returns:
        Dict with summary, classification_label, raw_response, model, timestamp

    Raises:
        Exception: On any parsing failure (fail-fast)
    """
    if response.status_code != 200:
        raise Exception(
            f"LLM API failed with status {response.status_code}: {response.text}"
        )

    try:
        response_json = response.json()
    except Exception as e:
        raise Exception(f"Could not decode LLM API response as JSON: {e}")

    # Extract content string from message
    content = response_json.get("message", {}).get("content", "")
    if not content:
        raise Exception("Empty content in LLM response")

    # Try to parse content as JSON
    parsed_content = None
    try:
        parsed_content = json.loads(content)
    except json.JSONDecodeError:
        # Try to extract JSON block from markdown fence
        if "```json" in content:
            json_start = content.find("```json") + 7
            json_end   = content.find("```", json_start)
            if json_end > json_start:
                try:
                    parsed_content = json.loads(content[json_start:json_end].strip())
                except json.JSONDecodeError as e:
                    raise Exception(f"Could not parse JSON inside markdown block: {e}")
            else:
                raise Exception("Malformed markdown JSON block in LLM response")
        else:
            raise Exception(
                f"LLM response content is not valid JSON and has no markdown fence: {content[:200]}"
            )

    # Validate required fields
    summary        = parsed_content.get("summary", "")
    classification = parsed_content.get("classification_label", "")

    if not summary or not classification:
        raise Exception(
            f"LLM response missing required fields. Got keys: {list(parsed_content.keys())}"
        )

    if classification not in classification_labels:
        log.warning(
            "LLM returned classification '%s' which is not in the expected label list",
            classification,
        )

    return {
        "summary":              summary,
        "classification_label": classification,
        "raw_response":         response_json,
        "model":                response_json.get("model", MODEL),
        "timestamp":            datetime.datetime.now().isoformat(),
    }


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def call_ollama(prompt: str, config) -> Dict[str, Any]:
    """
    Send a prompt to the remote LLM service and return parsed summary/classification.

    Uses the same endpoint, API key, model, and request format as the main
    pdf_extraction app (call_llama_api.py).

    Args:
        prompt:  Formatted prompt string built by prompt_bridge
        config:  DotsConfig instance (used for classification label validation)

    Returns:
        Dict with keys: summary, classification_label, raw_response, model, timestamp

    Raises:
        Exception: On any API or parse failure
    """
    log.info(
        "Calling LLM API (%s, model=%s, ~%d words in prompt)",
        API_URL,
        MODEL,
        len(prompt.split()),
    )

    try:
        response = _send_prompt(prompt, api_key=API_KEY)
    except Exception as e:
        raise Exception(f"LLM API request failed: {e}") from e

    return _parse_llm_response(response, config.CLASSIFICATION_LABELS)
