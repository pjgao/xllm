#!/usr/bin/env bash
set -euo pipefail

# Reproduce the Qwen3.5-35B-A3B four-die serving topology used for the
# open-loop QPS=4 measurement: two independent TP2 replicas behind the
# remaining-work proxy.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)

MODEL_PATH=${MODEL_PATH:-/home/data/weights/Qwen3.5-35B-A3B}
DRAFT_MODEL_PATH=${DRAFT_MODEL_PATH:-/home/data/weights/Qwen3.5-35B-A3B-mtp}
RUN_TAG=${RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}
RUN_DIR=${RUN_DIR:-"${REPO_ROOT}/repro_runs/qwen35_4die_qps4_service_${RUN_TAG}"}

REPLICA_A_DEVICES=${REPLICA_A_DEVICES:-10,11}
REPLICA_B_DEVICES=${REPLICA_B_DEVICES:-14,15}
REPLICA_A_CPU0=${REPLICA_A_CPU0:-400-439}
REPLICA_A_CPU1=${REPLICA_A_CPU1:-440-479}
REPLICA_B_CPU0=${REPLICA_B_CPU0:-560-599}
REPLICA_B_CPU1=${REPLICA_B_CPU1:-600-639}

REPLICA_A_PORT0=${REPLICA_A_PORT0:-19340}
REPLICA_A_PORT1=${REPLICA_A_PORT1:-19341}
REPLICA_B_PORT0=${REPLICA_B_PORT0:-19350}
REPLICA_B_PORT1=${REPLICA_B_PORT1:-19351}
REPLICA_A_HCCL_PORT=${REPLICA_A_HCCL_PORT:-38380}
REPLICA_B_HCCL_PORT=${REPLICA_B_HCCL_PORT:-38390}
PROXY_PORT=${PROXY_PORT:-19415}
READY_TIMEOUT_SECONDS=${READY_TIMEOUT_SECONDS:-1800}

if [[ -z "${BINARY:-}" ]]; then
  for candidate in \
    "${REPO_ROOT}/build/lib.linux-aarch64-cpython-311/xllm/xllm" \
    "${REPO_ROOT}/build/xllm/core/server/xllm"; do
    if [[ -x "${candidate}" ]]; then
      BINARY=${candidate}
      break
    fi
  done
fi

if [[ -z "${BINARY:-}" || ! -x "${BINARY}" ]]; then
  echo "xLLM binary not found. Set BINARY to the executable built from this branch." >&2
  exit 1
fi
if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "Model directory does not exist: ${MODEL_PATH}" >&2
  exit 1
fi
if [[ ! -d "${DRAFT_MODEL_PATH}" ]]; then
  echo "Draft model directory does not exist: ${DRAFT_MODEL_PATH}" >&2
  exit 1
fi
if ! command -v python3 >/dev/null; then
  echo "python3 is required to run the replica proxy." >&2
  exit 1
fi
if ! command -v curl >/dev/null; then
  echo "curl is required for service readiness checks." >&2
  exit 1
fi

mkdir -p "${RUN_DIR}/replica_a" "${RUN_DIR}/replica_b" "${RUN_DIR}/proxy"
: >"${RUN_DIR}/pids.txt"

common_args=(
  --model "${MODEL_PATH}"
  --dp_size=1
  --ep_size=1
  --max_memory_utilization=0.8
  --max_tokens_per_batch=32768
  --max_tokens_per_chunk_for_prefill=8192
  --max_seqs_per_batch=16
  --block_size=128
  --communication_backend=hccl
  --enable_prefix_cache=false
  --enable_chunked_prefill=true
  --max_concurrent_requests=1024
  --enable_schedule_overlap=true
  --enable_graph=true
  --enable_shm=true
  --input_shm_size=512
  --output_shm_size=128
  --task=generate
  --backend=llm
  --draft_model "${DRAFT_MODEL_PATH}"
  --num_speculative_tokens=4
  --enable_mtp_draft_body_tp1=true
  --enable_graph_double_buffer=true
  --enable_graph_mode_decode_no_padding=true
  --random_seed=20260903
  --enable_fia_decode=true
  --enable_flashcomm1=false
  --enable_mmrs_fusion=false
  --expert_parallel_degree=0
  --enable_mega_moe=false
)

pids=()

launch_rank() {
  local devices=$1
  local cpu_range=$2
  local port=$3
  local hccl_port=$4
  local node_rank=$5
  local log_file=$6
  local bind_command=()

  if [[ -n "${cpu_range}" ]]; then
    if ! command -v numactl >/dev/null; then
      echo "numactl is required when a CPU range is configured." >&2
      exit 1
    fi
    bind_command=(numactl -C "${cpu_range}")
  fi

  env \
    HCCL_OP_EXPANSION_MODE=AIV \
    HCCL_DETERMINISTIC=false \
    ASCEND_RT_VISIBLE_DEVICES="${devices}" \
    setsid nohup "${bind_command[@]}" "${BINARY}" "${common_args[@]}" \
      --port="${port}" \
      --master_node_addr="127.0.0.1:${hccl_port}" \
      --nnodes=2 \
      --node_rank="${node_rank}" \
      >>"${log_file}" 2>&1 &
  pids+=("$!")
  printf '%s\n' "$!" >>"${RUN_DIR}/pids.txt"
}

wait_for_backend() {
  local port=$1
  local deadline=$((SECONDS + READY_TIMEOUT_SECONDS))

  while ((SECONDS < deadline)); do
    if curl --fail --silent --output /dev/null "http://127.0.0.1:${port}/health"; then
      return 0
    fi
    for pid in "${pids[@]}"; do
      if ! kill -0 "${pid}" 2>/dev/null; then
        echo "xLLM process ${pid} exited before the service became ready." >&2
        return 1
      fi
    done
    sleep 2
  done

  echo "Timed out waiting for xLLM on port ${port}." >&2
  return 1
}

launch_rank "${REPLICA_A_DEVICES}" "${REPLICA_A_CPU0}" \
  "${REPLICA_A_PORT0}" "${REPLICA_A_HCCL_PORT}" 0 \
  "${RUN_DIR}/replica_a/node_0.log"
launch_rank "${REPLICA_A_DEVICES}" "${REPLICA_A_CPU1}" \
  "${REPLICA_A_PORT1}" "${REPLICA_A_HCCL_PORT}" 1 \
  "${RUN_DIR}/replica_a/node_1.log"
launch_rank "${REPLICA_B_DEVICES}" "${REPLICA_B_CPU0}" \
  "${REPLICA_B_PORT0}" "${REPLICA_B_HCCL_PORT}" 0 \
  "${RUN_DIR}/replica_b/node_0.log"
launch_rank "${REPLICA_B_DEVICES}" "${REPLICA_B_CPU1}" \
  "${REPLICA_B_PORT1}" "${REPLICA_B_HCCL_PORT}" 1 \
  "${RUN_DIR}/replica_b/node_1.log"

echo "Waiting for both TP2 replicas to become ready..."
wait_for_backend "${REPLICA_A_PORT0}"
wait_for_backend "${REPLICA_B_PORT0}"

setsid nohup python3 "${REPO_ROOT}/scripts/perf/tp_replica_proxy.py" \
  --listen-port "${PROXY_PORT}" \
  --backend "http://127.0.0.1:${REPLICA_A_PORT0}" \
  --backend "http://127.0.0.1:${REPLICA_B_PORT0}" \
  --service-time-ms 320 \
  --release-on-done \
  --routing-policy remaining-work \
  --log-file "${RUN_DIR}/proxy/requests.jsonl" \
  >>"${RUN_DIR}/proxy/proxy.log" 2>&1 &
proxy_pid=$!
printf '%s\n' "${proxy_pid}" >>"${RUN_DIR}/pids.txt"

proxy_deadline=$((SECONDS + 30))
until curl --fail --silent --output /dev/null \
  "http://127.0.0.1:${PROXY_PORT}/health"; do
  if ! kill -0 "${proxy_pid}" 2>/dev/null; then
    echo "Replica proxy exited during startup; see ${RUN_DIR}/proxy/proxy.log." >&2
    exit 1
  fi
  if ((SECONDS >= proxy_deadline)); then
    echo "Timed out waiting for replica proxy on port ${PROXY_PORT}." >&2
    exit 1
  fi
  sleep 1
done

echo "Service ready: http://127.0.0.1:${PROXY_PORT}/v1/chat/completions"
echo "Logs and PIDs: ${RUN_DIR}"
echo "Stop all processes with: kill \$(cat '${RUN_DIR}/pids.txt')"
