"""
Test script for DOTS OCR pipeline
=================================

Simple test to verify the pipeline works end-to-end with placeholder backend.
"""

import logging
from pathlib import Path
import sys

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

log = logging.getLogger("test")


def test_pipeline():
    """Test the pipeline with a placeholder backend."""
    
    log.info("=" * 60)
    log.info("DOTS OCR Pipeline Test")
    log.info("=" * 60)
    
    try:
        from config import get_config
        from pipeline import run_pipeline
        from call_llama_api import API_URL, MODEL

        # Get config and check device
        config = get_config()
        effective_device = config.get_effective_device()

        log.info(f"\nConfiguration:")
        log.info(f"  Device policy:     {config.DEVICE_POLICY}")
        log.info(f"  Effective device:  {effective_device}")
        log.info(f"  Page cap:          {config.PAGE_CAP}")
        log.info(f"  DPI:               {config.DPI}")
        log.info(f"  Performance preset:{config.PERFORMANCE_PRESET}")
        log.info(f"  LLM endpoint:      {API_URL}")
        log.info(f"  LLM model:         {MODEL}")
        
        # Create a dummy PDF for testing (or check if one exists)
        test_pdf = Path("test_sample.pdf")
        
        if not test_pdf.exists():
            log.warning(f"\nTest PDF not found: {test_pdf}")
            log.warning("To fully test the pipeline, provide a PDF file named 'test_sample.pdf'")
            log.warning("\nFor now, testing with placeholder backend only...")
            
            # Test without actual PDF by importing modules
            from extractor import DotsExtractor
            from prompt_bridge import PromptBridge
            from ollama_client import OllamaClient
            
            log.info("\n✅ All modules imported successfully")
            log.info("✅ Config loaded successfully")
            log.info("✅ Device detection working")
            
            # Test extractor initialization
            extractor = DotsExtractor(config)
            log.info(f"✅ DotsExtractor initialized (device: {extractor.effective_device})")
            
            # Test prompt bridge initialization
            bridge = PromptBridge(config)
            log.info(f"✅ PromptBridge initialized (max words: {bridge.max_words})")
            
            # Test Ollama client initialization
            client = OllamaClient(config)
            log.info(f"✅ OllamaClient initialized (model: {client.model})")
            
            log.info("\n" + "=" * 60)
            log.info("MODULE TEST PASSED")
            log.info("=" * 60)
            log.info("\nTo test full pipeline:")
            log.info("1. Place a PDF file named 'test_sample.pdf' in this directory")
            log.info("2. Ensure LLAMA_API_KEY is set (or check call_llama_api.py)")
            log.info("3. Run this test again")
            
            return True
        
        log.info(f"\n📄 Processing test PDF: {test_pdf}")
        
        # Run the full pipeline
        result = run_pipeline(test_pdf)
        
        if result["status"] == "success":
            log.info("\n" + "=" * 60)
            log.info("PIPELINE TEST PASSED")
            log.info("=" * 60)
            log.info(f"\n📊 Results:")
            log.info(f"  Classification: {result['classification']}")
            log.info(f"  Summary: {result['summary'][:100]}...")
            log.info(f"\n⏱️  Processing time: {result['metadata']['processing_time_seconds']:.2f}s")
            log.info(f"  Device used: {result['metadata']['device_used']}")
            log.info(f"\n📁 Artifacts saved:")
            for artifact_type, path in result["artifacts"].items():
                log.info(f"  - {artifact_type}: {path}")
            
            if result['metadata']['warnings']:
                log.info(f"\n⚠️  Warnings: {result['metadata']['warnings']}")
            
            return True
        else:
            log.error(f"\n❌ Pipeline failed: {result.get('error', 'Unknown error')}")
            return False
            
    except ImportError as e:
        log.error(f"\n❌ Import error: {e}")
        log.error("\nMake sure you've installed requirements:")
        log.error("  pip install -r requirements.txt")
        return False
    
    except Exception as e:
        log.error(f"\n❌ Test failed: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    success = test_pipeline()
    sys.exit(0 if success else 1)
