#!/usr/bin/env bash
# One-hit launcher: activate venv and start the DOTS OCR Gradio app.
# Model loads in-process on first PDF upload (no separate server needed).
# Run from pdf_extraction_dots_ocr: bash launch.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

GRADIO_PORT="${GRADIO_PORT:-7862}"
DOTS_MODEL_PATH="${DOTS_MODEL_PATH:-rednote-hilab/dots.ocr}"

if [[ ! -d venv_dot_ocr ]]; then
  echo "venv_dot_ocr not found. Run: bash setup_venv.sh"
  exit 1
fi

source venv_dot_ocr/bin/activate

export DOTS_MODEL_PATH

echo "Launching DOTS OCR Gradio app..."
echo "  Model:  $DOTS_MODEL_PATH"
echo "  Port:   $GRADIO_PORT"
echo "  The DOTS model loads on the first PDF upload (no separate server needed)."
echo ""

exec python gradio_app.py --port "$GRADIO_PORT"
