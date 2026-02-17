"""
Prompt Bridge Module
===================

Transforms normalized DOTS JSON into custom Ollama prompts for summary/classification.
"""

import logging
from typing import Dict, Any, List
import textwrap

log = logging.getLogger("prompt_bridge")


class PromptBridge:
    """
    Converts normalized DOTS extraction into Ollama-ready prompts.
    
    Key responsibilities:
    - Format OCR text, layout categories, tables, formulas
    - Preserve reading order
    - Manage prompt size with truncation strategy
    - Inject classification labels
    """
    
    def __init__(self, config):
        """
        Initialize prompt bridge with configuration.
        
        Args:
            config: DotsConfig instance
        """
        self.config = config
        self.max_words = config.MAX_PROMPT_WORDS
        self.classification_labels = config.CLASSIFICATION_LABELS
    
    def build_prompt(self, normalized_extraction: Dict[str, Any]) -> str:
        """
        Build Ollama prompt from normalized extraction.
        
        Args:
            normalized_extraction: Normalized DOTS extraction result
            
        Returns:
            Formatted prompt string ready for Ollama
        """
        log.info("Building Ollama prompt from DOTS extraction")
        
        # Extract metadata and elements
        metadata = normalized_extraction.get("file_metadata", {})
        elements = normalized_extraction.get("elements", [])
        warnings = normalized_extraction.get("warnings", [])
        
        # Build prompt header with instructions
        prompt = self._build_prompt_header()
        
        # Add document context
        prompt += self._build_document_context(metadata)
        
        # Add extracted content in reading order
        content_section = self._build_content_section(elements)
        
        # Check word count and truncate if needed
        current_words = len(prompt.split())
        content_words = len(content_section.split())
        
        if current_words + content_words > self.max_words:
            log.warning(f"Prompt would exceed {self.max_words} words, truncating content")
            available_words = self.max_words - current_words - 50  # Reserve for truncation notice
            content_section = self._truncate_content(content_section, available_words)
        
        prompt += content_section
        
        # Add warnings if present
        if warnings:
            prompt += f"\n\n[Extraction Warnings: {'; '.join(warnings)}]\n"
        
        log.info(f"Built prompt with {len(prompt.split())} words from {len(elements)} elements")
        
        return prompt
    
    def _build_prompt_header(self) -> str:
        """
        Build prompt header with task instructions.

        Note: call_llama_api.py already injects a system message that says
        "Return ONLY the JSON requested by the user", so we only need to state
        the user-level task and schema here.
        """
        labels_str = ", ".join(self.classification_labels)

        return textwrap.dedent(f"""\
            You are an expert document analyst specialising in scanned and digitised documents.

            TASKS:
            1. Summarise the document in 2-3 crisp sentences that capture the key information.
            2. Classify the document into exactly ONE category from the list below.

            ALLOWED CATEGORIES:
            {labels_str}

            Return ONLY valid JSON — no markdown, no extra commentary:
            {{
              "summary": "2-3 sentence summary here",
              "classification_label": "one_category_from_list_above"
            }}

            DOCUMENT CONTENT:

            """)
    
    def _build_document_context(self, metadata: Dict[str, Any]) -> str:
        """Build document context section from metadata."""
        filename = metadata.get("filename", "unknown")
        total_pages = metadata.get("total_pages", "unknown")
        processed_pages = metadata.get("processed_pages", "unknown")
        method = metadata.get("extraction_method", "unknown")
        
        return f"""[Document: {filename}]
[Total Pages: {total_pages}, Processed: {processed_pages}]
[Extraction Method: {method}]

"""
    
    def _build_content_section(self, elements: List[Dict[str, Any]]) -> str:
        """
        Build content section from elements in reading order.
        
        Args:
            elements: List of normalized elements
            
        Returns:
            Formatted content string
        """
        # Sort by order
        sorted_elements = sorted(elements, key=lambda e: e.get("order", 999))
        
        content_parts = []
        
        for elem in sorted_elements:
            elem_type = elem.get("element_type", "unknown")
            category = elem.get("category", "unknown")
            content = elem.get("content", "")
            page_num = elem.get("page_number", "?")
            confidence = elem.get("confidence", 0.0)
            
            # Format based on type and category
            if elem_type == "text":
                if category == "heading":
                    formatted = f"\n[Page {page_num} - HEADING]\n{content}\n"
                elif category == "paragraph":
                    formatted = f"\n[Page {page_num} - Paragraph]\n{content}\n"
                else:
                    formatted = f"\n[Page {page_num} - Text ({category})]\n{content}\n"
            
            elif elem_type == "table":
                # For tables, show simplified representation
                formatted = f"\n[Page {page_num} - TABLE]\n"
                formatted += f"Table content (HTML): {self._simplify_table(content)}\n"
            
            elif elem_type == "formula":
                formatted = f"\n[Page {page_num} - FORMULA]\n"
                formatted += f"Formula (LaTeX): {content}\n"
            
            else:
                formatted = f"\n[Page {page_num} - {category}]\n{content}\n"
            
            # Add confidence note for low confidence
            if confidence > 0 and confidence < 0.8:
                formatted += f"  [OCR confidence: {confidence:.2f}]\n"
            
            content_parts.append(formatted)
        
        return "".join(content_parts)
    
    def _simplify_table(self, html_content: str) -> str:
        """
        Simplify HTML table for prompt readability.
        
        For now, just return truncated HTML. Later could parse and format as text.
        """
        if len(html_content) > 300:
            return html_content[:300] + "... [table truncated]"
        return html_content
    
    def _truncate_content(self, content: str, max_words: int) -> str:
        """
        Truncate content to fit within word limit.
        
        Args:
            content: Content string to truncate
            max_words: Maximum words to keep
            
        Returns:
            Truncated content with notice
        """
        words = content.split()
        if len(words) <= max_words:
            return content
        
        truncated_words = words[:max_words]
        truncated_content = " ".join(truncated_words)
        
        return truncated_content + f"\n\n[Content truncated: {len(words)} words reduced to {max_words} words]"


def build_ollama_prompt(normalized_extraction: Dict[str, Any], config) -> str:
    """
    Convenience function to build Ollama prompt.
    
    Args:
        normalized_extraction: Normalized DOTS extraction result
        config: DotsConfig instance
        
    Returns:
        Formatted prompt string
    """
    bridge = PromptBridge(config)
    return bridge.build_prompt(normalized_extraction)
