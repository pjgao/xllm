#!/usr/bin/env bash
set -euo pipefail

# Open-loop Poisson QPS=4 workload used for the approximately 3m20s formal run.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)

MODEL_NAME=${MODEL_NAME:-Qwen3.5-35B-A3B}
TOKENIZER_PATH=${TOKENIZER_PATH:-/home/data/weights/Qwen3.5-35B-A3B}
PROXY_HOST=${PROXY_HOST:-127.0.0.1}
PROXY_PORT=${PROXY_PORT:-19415}
RATE=${RATE:-4}
REQUESTS=${REQUESTS:-800}
WARMUP_REQUESTS=${WARMUP_REQUESTS:-10}
RUN_TAG=${RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}
OUTPUT_DIR=${OUTPUT_DIR:-"${REPO_ROOT}/repro_runs/qwen35_4die_qps4_eval_${RUN_TAG}"}

if ! command -v evalscope >/dev/null; then
  echo "evalscope is not installed or is not available on PATH." >&2
  exit 1
fi
if ! command -v curl >/dev/null; then
  echo "curl is required for the service readiness check." >&2
  exit 1
fi
if [[ ! -d "${TOKENIZER_PATH}" ]]; then
  echo "Tokenizer directory does not exist: ${TOKENIZER_PATH}" >&2
  exit 1
fi
if ! curl --fail --silent --output /dev/null \
  "http://${PROXY_HOST}:${PROXY_PORT}/health"; then
  echo "Service proxy is not ready on ${PROXY_HOST}:${PROXY_PORT}." >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

evalscope perf \
  --model "${MODEL_NAME}" \
  --api openai \
  --url "http://${PROXY_HOST}:${PROXY_PORT}/v1/chat/completions" \
  --tokenizer-path "${TOKENIZER_PATH}" \
  --dataset random \
  --dataset-offset 0 \
  --parallel -1 \
  --rate "${RATE}" \
  --open-loop \
  --number "${REQUESTS}" \
  --warmup-num "${WARMUP_REQUESTS}" \
  --max-prompt-length 2048 \
  --min-prompt-length 2048 \
  --max-tokens 40 \
  --min-tokens 40 \
  --seed 20260904 \
  --temperature 0 \
  --extra-args '{"ignore_eos":true}' \
  --outputs-dir "${OUTPUT_DIR}"
