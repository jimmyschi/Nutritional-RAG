#!/usr/bin/env bash
set -euo pipefail

API_BASE_URL="${API_BASE_URL:-http://127.0.0.1:8000}"
METRICS_URL="${METRICS_URL:-http://127.0.0.1:8000/metrics}"
EVAL_SET="${EVAL_SET:-data/eval/nutrition_eval_set.ndjson}"
TOP_K="${TOP_K:-5}"
RERANK_MULTIPLIER="${RERANK_MULTIPLIER:-3}"
ROUNDS="${ROUNDS:-120}"
SLEEP_SECONDS="${SLEEP_SECONDS:-0.3}"
CONCURRENCY="${CONCURRENCY:-12}"
BURSTS="${BURSTS:-4}"

if [[ ! -f "scripts/simulate_traffic.py" ]]; then
  echo "Run this script from the project root."
  exit 1
fi

echo "[demo] Warm cache and sustain traffic for dashboard stats"
python scripts/simulate_traffic.py \
  --api-base-url "$API_BASE_URL" \
  --metrics-url "$METRICS_URL" \
  --eval-set "$EVAL_SET" \
  --rounds "$ROUNDS" \
  --sleep-seconds "$SLEEP_SECONDS" \
  --top-k "$TOP_K" \
  --rerank-candidate-multiplier "$RERANK_MULTIPLIER" \
  --use-cache

echo "[demo] Run short concurrent bursts to raise in-flight queries"
for burst in $(seq 1 "$BURSTS"); do
  printf '{"question":"How are carbohydrates used in endurance exercise?","top_k":%s,"rerank_candidate_multiplier":%s,"use_cache":false,"generate_answer":false}\n' "$TOP_K" "$RERANK_MULTIPLIER" \
    | xargs -I{} -P "$CONCURRENCY" sh -c "curl -s -o /dev/null -X POST '$API_BASE_URL/query' -H 'Content-Type: application/json' -d '{}'"
  sleep 1
done

echo "[demo] Done. Open Grafana and set dashboard time range to Last 1 hour."
