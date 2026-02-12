# PDF Extraction & LLM Analysis Pipeline

A comprehensive PDF processing system that extracts text and images from PDF documents, then uses Large Language Models (LLM) to automatically classify and summarize the content. Features a modern web interface built with Gradio for easy document processing.

## 🌟 Features

### Core Capabilities
- **Multi-Method PDF Extraction**: Combines Docling, pikepdf for robust text and image extraction
- **Image Processing**: Extracts and captions images using BLIP (Salesforce's image captioning model)
- **Image OCR**: Extracts text from images using PaddleOCR (with EasyOCR fallback)
- **Smart Image Filtering**: Page-relative and absolute size thresholds to focus on meaningful images
- **LLM Analysis**: Integrates with Ollama/LLaMA API for intelligent document classification
- **Document Classification**: Automatically categorizes documents into 20+ types (invoice, resume, scientific paper, etc.)
- **Web Interface**: Interactive Gradio UI with OCR controls for uploading and processing multiple PDFs
- **Batch Processing**: Handle multiple documents with session-based caching
- **Kubernetes Integration**: Production-ready deployment with auto-scaling via cron jobs

### Supported Document Types
The system can classify documents into:
- **Financial**: invoice, budget, financial_statement, tax_record, bank_card
- **Legal**: lease_agreement, property_deed, insurance_document, vehicle_title
- **Business**: memo, letter, email, presentation, specification, form
- **Academic**: scientific_document, resume, questionnaire
- **Media**: news_article, advertisement
- **Other**: ID, handwritten, unclassified

## 🏗️ Architecture

```
PDF Input → Docling Extraction → Text Cleaning → JSON Generation
                ↓
        Image Extraction (pikepdf)
        ├─ Page-relative size filtering
        └─ Absolute size filtering
                ↓
        Parallel Processing:
        ├─ Image Captioning (BLIP)
        └─ Text Extraction (PaddleOCR/EasyOCR)
                ↓
        Combined Image Data (caption + OCR text)
                ↓
        LLM Processing (Ollama/LLaMA)
                ↓
        Classification + Summary → Structured JSON Output
```

**OCR Trigger**: Image OCR is performed when:
- Document has fewer than `MIN_WORDS` (default: 200) extracted by Docling
- Images meet both absolute (`MIN_IMAGE_AREA_PIXELS`) and relative (`MIN_IMAGE_AREA_PERCENT`) size thresholds
- OCR is enabled in configuration or UI

## 📋 Prerequisites

- **Python**: 3.13+ (Python 3.10+ should work)
- **OS**: Windows 10/11, macOS, or Linux
- **Optional**: CUDA-capable GPU for faster image processing
- **Kubernetes**: For production deployment (optional)

## 🚀 Quick Start

### 1. Clone and Setup

```bash
# Clone the repository
git clone <repository-url>
cd pdf_extraction

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows PowerShell:
.\venv\Scripts\Activate.ps1
# Windows CMD:
.\venv\Scripts\activate.bat
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Access

```bash
# Set your LLM API key
export LLAMA_API_KEY="your-api-key-here"

# Or edit call_llama_api.py to set API_KEY directly
```

### 3. Run the Application

#### Web Interface (Recommended)
```bash
python pdf_extraction/gradio_app.py

# With custom port
python pdf_extraction/gradio_app.py --port 7860

# With Gradio share link
python pdf_extraction/gradio_app.py --share
```
Then open http://localhost:7861 in your browser.

**Using OCR in the Web UI:**
1. Click "⚙️ Extraction Options" to expand OCR settings
2. Check "Enable Image OCR" to extract text from images
3. Select OCR language (English, Chinese, French, German, Spanish, etc.)
4. Upload PDF(s) - OCR will be applied to images automatically
5. Results show both visual descriptions (BLIP) and extracted text (OCR)

#### Command Line Processing
```bash
# Process a single PDF
python pdf_extraction/pdf_pipeline.py document.pdf output/

# Process all PDFs in a directory
python pdf_extraction/pdf_pipeline.py /path/to/pdfs/ output/

# Overwrite existing results
python pdf_extraction/pdf_pipeline.py document.pdf output/ --overwrite
```

#### Test LLM API Connection
```bash
python pdf_extraction/call_llama_api.py "Say hello."
```

## 📁 Project Structure

```
pdf_extraction/
├── pdf_extraction/
│   ├── pdf_extractor.py          # PDF extraction logic (Docling + pikepdf + OCR)
│   ├── pdf_pipeline.py            # Main orchestrator for the full pipeline
│   ├── call_llama_api.py          # LLM API wrapper (Ollama/LLaMA)
│   ├── gradio_app.py              # Web UI (Gradio-based)
│   ├── gradio_app_light_demo.py   # Lightweight demo version
│   ├── requirements.txt           # Python dependencies
│   ├── docs/
│   │   ├── confluence_documentation.pdf  # Project documentation
│   │   ├── deployment-full.yaml          # Kubernetes deployment
│   │   ├── tolerations-patch.yaml        # K8s tolerations config
│   │   ├── k8s-config                    # Kubernetes configuration
│   │   ├── k8s-sip-eks-mcap-use-dev-default-conf  # EKS config
│   │   ├── OLLAMA_SERVICE_COMMANDS.md    # Ollama service management
│   │   └── Notes about LLM Cron jobs.txt # Cron job documentation
│   └── output/                    # Generated outputs (gitignored)
│       ├── *_dual.json            # Extracted text + images
│       ├── *_prompt_response.json # LLM responses
│       └── llama_responses.json   # Aggregate log
├── .gitignore
└── README.md
```

## 🔧 Configuration

### PDF Extraction Settings
Edit settings in `pdf_extraction/config.py`:
```python
# Text extraction
MIN_WORDS = 200                    # If text < threshold, extract images too
MAX_PROMPT_WORDS = 1000            # Max words in LLM prompt

# Image extraction
MAX_IMAGES = 5                     # Maximum images to extract per PDF
MIN_IMAGE_AREA_PIXELS = 50_000     # Absolute minimum (pixels)
MIN_IMAGE_AREA_PERCENT = 0.15      # Minimum 15% of page area

# OCR settings
ENABLE_IMAGE_OCR = True            # Enable/disable OCR
OCR_ENGINE = "paddleocr"           # OCR engine to use
OCR_LANGUAGES = ["en"]             # OCR language(s)
MAX_OCR_DIMENSION = 2000           # Max image dimension for OCR (memory safety)
```

**Configuration Overrides**: The Gradio UI allows per-session overrides. CLI uses defaults from `config.py`.

### LLM API Settings
Edit `call_llama_api.py`:
```python
API_URL = "https://your-llm-endpoint/api/chat"
MODEL = "llama3.1:8b-instruct-q8_0"
API_KEY = "your-api-key"
```

### Prompt Configuration
Modify the prompt template in `pdf_pipeline.py` → `create_prompt()` function to customize:
- Classification categories
- Summary style and length
- Response format

## 🎯 Usage Examples

### Web Interface
1. Launch the Gradio app: `python pdf_extraction/gradio_app.py`
2. Upload PDF files (up to 5 at once)
3. View results:
   - Document summary and classification
   - Full extraction JSON
   - LLM prompt and response
   - PDF preview

### Python API
```python
from pathlib import Path
from pdf_extraction.pdf_pipeline import run

# Process PDFs programmatically
input_path = Path("document.pdf")
output_dir = Path("output")
run(input_path, output_dir, overwrite=False)
```

### Batch Processing
```bash
# Process all PDFs in a directory
python pdf_extraction/pdf_pipeline.py /path/to/pdfs/ output/

# Results will be aggregated in output/llama_responses.json
```

## 📊 Output Format

### Extraction JSON (`*_dual.json`)
```json
{
  "file_metadata": {
    "filename": "document.pdf",
    "total_pages": 5,
    "word_count": 1234,
    "extracted_at": "2025-02-04T10:30:00"
  },
  "elements": [
    {
      "element_id": "text_1",
      "element_type": "text",
      "content": "Document text...",
      "page_number": 1,
      "order": 1
    },
    {
      "element_id": "image_1",
      "element_type": "image",
      "caption": "Image description...",
      "ocr_text": "Text extracted from image...",
      "page_number": 2,
      "order": 2,
      "image_224_jpeg_base64": "...",
      "metadata": {
        "image_width": 800,
        "image_height": 600,
        "page_area_percentage": 45.2,
        "ocr_engine": "paddleocr"
      }
    }
  ]
}
```

### LLM Response (`*_prompt_response.json`)
```json
{
  "llm_prompt": {
    "content": "Full prompt sent to LLM..."
  },
  "response_body": {
    "model": "llama3.1:8b-instruct-q8_0",
    "content": {
      "summary": "Brief document summary...",
      "classification_label": "invoice"
    },
    "total_duration": 5234567890,
    "eval_count": 150
  }
}
```

## 🐳 Kubernetes Deployment

### Current Ollama Service Status
- **Namespace**: `ml-ollama`
- **Auto-scaling**: Managed by cron jobs
  - Scale up: Weekly (Monday 10:59 AM) or Monthly (23rd at 11:12 AM)
  - Scale down: Daily (7:02 PM) or Monthly (22nd at 11:18 AM)

### Manual Scaling
```bash
# Configure kubectl
export KUBECONFIG="path/to/k8s-config"
kubectl config use-context sip-eks-mcap-use-dev-default-eks_sip-eks-mcap-use-dev

# Scale up
kubectl -n ml-ollama scale deployment ml-ollama-ollama-gpu-cronjobs --replicas=1
kubectl -n ml-ollama scale deployment ml-ollama-ollama-gpu-cronjobs-auth --replicas=1

# Scale down
kubectl -n ml-ollama scale deployment ml-ollama-ollama-gpu-cronjobs --replicas=0
kubectl -n ml-ollama scale deployment ml-ollama-ollama-gpu-cronjobs-auth --replicas=0
```

### Deployment
```bash
# Apply deployment
kubectl apply -f pdf_extraction/docs/deployment-full.yaml

# Apply tolerations for GPU nodes
kubectl apply -f pdf_extraction/docs/tolerations-patch.yaml
```

## 🔒 Security & Corporate Environments

The system includes SSL bypass patches for corporate proxies and firewalls:
- Disables SSL verification for development
- Configures environment variables for model downloads
- Patches requests library for unverified connections

**⚠️ Warning**: Only use SSL bypass in trusted development environments. Remove these patches in production.

## 🛠️ Dependencies

### Core Libraries
- **PDF Processing**: pikepdf, docling
- **OCR**: paddleocr (primary), easyocr (fallback), paddlepaddle
- **ML/AI**: torch, transformers (BLIP model)
- **Web Framework**: gradio
- **HTTP Client**: requests, urllib3
- **Image Processing**: pillow (PIL)
- **Progress Tracking**: tqdm

See `requirements.txt` for exact versions.

**Note**: PaddleOCR and paddlepaddle are large dependencies (~400MB). First run downloads OCR models (~50MB). Subsequent runs use cached models.

## 🐛 Troubleshooting

### Common Issues

**Problem**: "Docling not installed" error
```bash
pip install docling>=0.7
```

**Problem**: BLIP model download fails (SSL errors)
```bash
# Check SSL bypass is enabled in gradio_app.py
# Or download models manually and point to local cache
```

**Problem**: LLM API returns 401 Unauthorized
```bash
# Verify API key is set correctly
echo $LLAMA_API_KEY
# or check call_llama_api.py → API_KEY constant
```

**Problem**: Out of memory during image processing
```python
# In config.py, reduce MAX_IMAGES or MAX_OCR_DIMENSION
MAX_IMAGES = 2              # Process fewer images
MAX_OCR_DIMENSION = 1500    # Resize larger images before OCR
```

**Problem**: PaddleOCR not installing or failing
```bash
# Try EasyOCR as fallback (already in requirements)
# In config.py:
OCR_ENGINE = "easyocr"

# Or install PaddleOCR manually
pip install paddleocr paddlepaddle --upgrade
```

**Problem**: OCR producing poor results
```python
# In Gradio UI or config.py:
# 1. Ensure correct language is selected
OCR_LANGUAGES = ["en"]  # or "ch", "fr", etc.

# 2. Check image quality - OCR works best on:
#    - High resolution images (DPI > 150)
#    - Clear, high-contrast text
#    - Non-rotated, non-skewed images
```

**Problem**: OCR is slow
```bash
# PaddleOCR: First run downloads models (~50MB)
# Subsequent runs use cached models and are much faster
# For GPU acceleration, install paddlepaddle-gpu instead of paddlepaddle
```

**Problem**: Kubernetes pods not running
```bash
# Check Ollama service is scaled up
kubectl -n ml-ollama get deployments
kubectl -n ml-ollama get pods
```

## 📝 Development

### Adding New Document Types
Edit `pdf_pipeline.py` → `create_prompt()` to add classifications:
```python
classification_categories = [
    "invoice", "resume", "scientific_document",
    "your_new_type",  # Add here
    # ... other types
]
```

### Customizing Extraction
Modify `pdf_extractor.py`:
- Adjust OCR settings
- Change image captioning model
- Customize text cleaning logic

### Modifying the UI
Edit `gradio_app.py`:
- Change max file uploads
- Customize display layout
- Add new tabs or visualizations

## 📄 License

[Specify your license here]

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📧 Contact

[Add contact information or support channels]

## 🙏 Acknowledgments

- **Docling**: For comprehensive PDF extraction
- **Salesforce BLIP**: For image captioning
- **Meta LLaMA**: For document analysis
- **Gradio**: For the web interface framework
