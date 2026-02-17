#!/usr/bin/env bash
# One-hit launcher: start vLLM server, wait for health, then run Gradio app.
# For Linux (e.g. jovyan). Run from pdf_extraction_dots_ocr: bash launch.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_MODEL="${VLLM_MODEL:-rednote-hilab/dots.ocr-1.5}"
GRADIO_PORT="${GRADIO_PORT:-7862}"
HEALTH_MAX_WAIT="${HEALTH_MAX_WAIT:-120}"
HEALTH_INTERVAL="${HEALTH_INTERVAL:-5}"

if [[ ! -d venv_dot_ocr ]]; then
  echo "venv_dot_ocr not found. Run: bash setup_venv.sh"
  exit 1
fi

source venv_dot_ocr/bin/activate

echo "Starting vLLM server (model=$VLLM_MODEL, port=$VLLM_PORT)..."
vllm serve "$VLLM_MODEL" \
  --served-model-name model \
  --port "$VLLM_PORT" \
  --trust-remote-code \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.85 &

VLLM_PID=$!

cleanup() {
  if kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "Stopping vLLM server (PID $VLLM_PID)..."
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT SIGINT SIGTERM

echo "Waiting for vLLM health (max ${HEALTH_MAX_WAIT}s, interval ${HEALTH_INTERVAL}s)..."
elapsed=0
while (( elapsed < HEALTH_MAX_WAIT )); do
  if curl -sf "http://localhost:${VLLM_PORT}/health" >/dev/null 2>&1; then
    echo "vLLM is ready."
    break
  fi
  sleep "$HEALTH_INTERVAL"
  elapsed=$(( elapsed + HEALTH_INTERVAL ))
done

if (( elapsed >= HEALTH_MAX_WAIT )); then
  echo "vLLM did not become healthy in ${HEALTH_MAX_WAIT}s. Exiting."
  exit 1
fi

export VLLM_PORT
export VLLM_HOST="${VLLM_HOST:-localhost}"

echo "Launching Gradio app on port $GRADIO_PORT..."
exec python gradio_app.py --port "$GRADIO_PORT"
