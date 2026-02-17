#!/usr/bin/env bash
# Setup script for DOTS OCR venv_dot_ocr on Linux

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Creating virtual environment: venv_dot_ocr"
python3 -m venv venv_dot_ocr

echo ""
echo "Activating virtual environment..."
source venv_dot_ocr/bin/activate

echo ""
echo "Upgrading pip..."
pip install --upgrade pip

echo ""
echo "Installing requirements..."
pip install -r requirements.txt

echo ""
echo "Setup complete!"
echo ""
echo "To activate this environment in the future:"
echo "  source venv_dot_ocr/bin/activate"
echo ""
echo "To run everything (vLLM + Gradio) in one hit:"
echo "  bash launch.sh"
echo ""
