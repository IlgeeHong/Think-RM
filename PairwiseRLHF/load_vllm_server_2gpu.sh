set -euo pipefail

export TORCH_NCCL_AVOID_RECORD_STREAMS="1"
export VLLM_USE_V1=0

### Path to the multiclass Think-GenRM
# MODEL="ilgee/hs2-naive-multiclass-max-ep5-lr5e-6-grpo-ep2-lr2e-6-kl1e-4-rollout512-half-v0"

### Path to the binary Think-GenRM model
MODEL="ilgee/hs2-naive-binary-max-ep5-lr1e-5-grpo-ep1-lr2e-6-kl1e-4-rollout512-v0"

### Directory to save vLLM server logs
LOG_DIR="/workspace/logs"

HEALTH_ENDPOINT="/v1/models"
TIMEOUT=180
RETRY_DELAY=5

# Each entry is a pair of GPU IDs
GPU_GROUPS=("0,1" "2,3" "4,5" "6,7")

declare -A PIDS

mkdir -p "$LOG_DIR"
rm -f "$LOG_DIR"/*.log

echo "Launching all shards (2 GPUs per shard)..."

for i in "${!GPU_GROUPS[@]}"; do
  GPUS=${GPU_GROUPS[$i]}
  PORT=$((8000 + i))
  LOGF="$LOG_DIR/vllm_server_${i}.log"
  echo "GPUs $GPUS -> port $PORT (logs -> $LOGF)"

  CUDA_VISIBLE_DEVICES=$GPUS \
    python3 -m vllm.entrypoints.openai.api_server \
      --model "$MODEL" \
      --tensor-parallel-size 2 \
      --port "$PORT" \
    >"$LOGF" 2>&1 &

  PIDS[$i]=$!
done

echo "Waiting $TIMEOUT seconds for shards to initialize..."
sleep "$TIMEOUT"

echo "Health checking shards..."
for i in "${!GPU_GROUPS[@]}"; do
  PORT=$((8000 + i))
  LOGF="$LOG_DIR/vllm_server_${i}.log"
  PID=${PIDS[$i]}

  HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:${PORT}${HEALTH_ENDPOINT} || echo "000")
  if [ "$HTTP_CODE" = "200" ]; then
    echo "Shard $i is healthy (HTTP 200)."
    continue
  fi

  echo "Shard $i failed health check (HTTP $HTTP_CODE). Restarting..."

  kill "$PID" 2>/dev/null || true

  until curl -s -o /dev/null -w "%{http_code}" http://localhost:${PORT}${HEALTH_ENDPOINT} | grep -q "^200$"; do
    echo "Restarting shard $i on GPUs $GPUS (port $PORT)..."
    CUDA_VISIBLE_DEVICES=$GPUS \
      python3 -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" \
        --tensor-parallel-size 2 \
        --port "$PORT" \
      >"$LOGF" 2>&1 &

    NEW_PID=$!
    PIDS[$i]=$NEW_PID
    sleep "$RETRY_DELAY"

    if ! kill -0 "$NEW_PID" 2>/dev/null; then
      echo "Process died, retrying in ${RETRY_DELAY}s..."
      continue
    fi
  done

  echo "Shard $i is now healthy."
done

echo "All 4 vLLM servers (2 GPUs each) are up and running!"
