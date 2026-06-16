#!/bin/bash
# Re-measure every needs_work entry in oracle_optimization_queue.csv with the
# standard methodology (fresh cache, CUDAGraph, GPU bench lock). Writes JSONL.
cd /tmp/scratch_space/better_benchmark
OUT=investigation_results/needs_work_remeasure_2026-06-09.jsonl
: > "$OUT"
ids=$(awk -F, '$3 ~ /needs_work/ {print $1}' investigation_results/oracle_optimization_queue.csv)
for id in $ids; do
  o=$(ls repros/canonical/$id/oracle_*.py 2>/dev/null | head -1)
  if [ -z "$o" ]; then
    echo "{\"repro_id\": \"$id\", \"status\": \"NO_ORACLE_FILE\"}" >> "$OUT"
    continue
  fi
  line=$(TORCHINDUCTOR_CACHE_DIR=$(mktemp -d) INDUCTOR_GPU_BENCH_LOCK=1 CUDA_VISIBLE_DEVICES=0 \
    timeout 600 python "$o" --bench 2>/dev/null | grep '^{' | tail -1)
  if [ -z "$line" ]; then
    echo "{\"repro_id\": \"$id\", \"status\": \"BENCH_FAILED\"}" >> "$OUT"
  else
    echo "$line" >> "$OUT"
  fi
done
echo "SWEEP_DONE $(wc -l < $OUT) entries"
