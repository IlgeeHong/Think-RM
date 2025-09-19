set -euo pipefail

export TORCH_NCCL_AVOID_RECORD_STREAMS="1"
export VLLM_USE_V1=0

### Path to the multiclass Think-RM
# MODEL="ilgee/Multiclass-Think-RM-8B"

### Path to the binary Think-RM
MODEL="ilgee/Binary-Think-RM-8B"

### Directory to save vLLM server logs
LOG_DIR="/workspace/logs"


HEALTH_ENDPOINT="/v1/models"
TIMEOUT=180 # seconds to wait before first health-check
RETRY_DELAY=5 # seconds between restart attempts
GPUS=(0 1 2 3 4 5 6 7)

declare -A PIDS
# --enforce-eager \
# 1) Prepare logs directory
mkdir -p "$LOG_DIR"
rm -f "$LOG_DIR"/*.log

############################################
# !!! Make sure chat template is valid !!! #
############################################

# 2) Launch all shards in background
echo "Launching all shards..."
for GPU in "${GPUS[@]}"; do
  PORT=$((8000 + GPU))
  LOGF="$LOG_DIR/vllm_server_${GPU}.log"
  echo "GPU $GPU -> port $PORT (logs -> $LOGF)"
  CUDA_VISIBLE_DEVICES=$GPU \
    python3 -m vllm.entrypoints.openai.api_server \
      --model "$MODEL" \
      --tensor-parallel-size 1 \
      --port "$PORT" \
    >"$LOGF" 2>&1 &
  PIDS[$GPU]=$!
done

echo "Waiting $TIMEOUT seconds for shards to initialize..."
sleep "$TIMEOUT"

# 3) Health-check & restart only failing shards
echo "Health checking shards..."
for GPU in "${GPUS[@]}"; do
  PORT=$((8000 + GPU))
  LOGF="$LOG_DIR/vllm_server_${GPU}.log"
  PID=${PIDS[$GPU]}

  HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:${PORT}${HEALTH_ENDPOINT} || echo "000")
  if [ "$HTTP_CODE" = "200" ]; then
    echo "Shard GPU $GPU is healthy (HTTP 200)."
    continue
  fi

  echo "Shard GPU $GPU failed health check (HTTP $HTTP_CODE). Restarting..."

  # kill the old process
  kill "$PID" 2>/dev/null || true

  # retry loop: keep restarting until healthy
  until curl -s -o /dev/null -w "%{http_code}" http://localhost:${PORT}${HEALTH_ENDPOINT} | grep -q "^200$"; do
    echo "Starting GPU $GPU on port $PORT (logs -> $LOGF)..."
    CUDA_VISIBLE_DEVICES=$GPU \
      python3 -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" \
        --tensor-parallel-size 1 \
        --port "$PORT" \
      >"$LOGF" 2>&1 &
    NEW_PID=$!
    PIDS[$GPU]=$NEW_PID

    # give it a bit before re-checking
    sleep "$RETRY_DELAY"

    # if the process died, echo and retry immediately
    if ! kill -0 "${PIDS[$GPU]}" 2>/dev/null; then
      echo "Process died, retrying in ${RETRY_DELAY}s..."
      continue
    fi
  done

  echo "Shard GPU $GPU is now healthy."
done

echo "All shards up and running!"

#sudo apt-get update
#sudo apt-get install psmisc
#sudo apt-get install lsof
#lsof -i :8000
#fuser -k 8000/tcp