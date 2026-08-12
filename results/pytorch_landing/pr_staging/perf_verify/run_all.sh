#!/bin/bash
# Sequential A/B perf verification driver. One bench_parallel invocation per (arm, set).
# Arms differ ONLY in PYTHONPATH (shadow worktree) + TORCHINDUCTOR_CACHE_DIR.
set -u
cd /tmp/scratch_space/better_benchmark
OUT=/tmp/scratch_space/pr_perf_verify
WT=/tmp/pt-verify-wt

run() {
  local arm="$1"; shift
  local tag="$1"; shift
  local outfile="$OUT/${tag}__${arm}.json"
  if [ -s "$outfile" ]; then echo "SKIP $tag $arm (exists)"; return 0; fi
  echo "=== RUN $tag arm=$arm $(date +%T) ==="
  PYTHONPATH=$WT/$arm TORCHINDUCTOR_CACHE_DIR=$OUT/cache_$arm \
    python scripts/bench_parallel.py "$@" --no-cd --output "$outfile" \
    > "$OUT/log_${tag}__${arm}.txt" 2>&1
  echo "=== DONE $tag arm=$arm rc=$? $(date +%T) ==="
}

SET_RSQRT=(repros/canonical/pointwise_2c331ef4f17f repros/canonical/pointwise_9128d8745e42 repros/canonical/mean_3840584eef9a)
SET_OS=(repros/canonical/amax_sum_02064a1e60ac repros/canonical/amax_sum_sum_04ddf882ff17 repros/canonical/amax_sum_sum_3f000d9caa57)
SET_CE=(repros/canonical/amax_sum_sum_42988b64e7f9 repros/canonical/amax_sum_sum_4bf8a79efec4)

# 1. rsqrt
run base  rsqrt_set "${SET_RSQRT[@]}" --all-shapes --gpus 0,1,2,3
run rsqrt rsqrt_set "${SET_RSQRT[@]}" --all-shapes --gpus 0,1,2,3
# 2. online-softmax
run base           os_set "${SET_OS[@]}" --all-shapes --gpus 0,1,2,3
run online-softmax os_set "${SET_OS[@]}" --all-shapes --gpus 0,1,2,3
# 3. ce-hoist corpus set
run base     ce_set "${SET_CE[@]}" --all-shapes --gpus 0,1,2,3
run ce-hoist ce_set "${SET_CE[@]}" --all-shapes --gpus 0,1,2,3
# 4. genai CrossEntropyForward full graph
run base       genai_ce --full-graphs repros/models/genai/static/CrossEntropyForward --gpus 0
run ce-hoist   genai_ce --full-graphs repros/models/genai/static/CrossEntropyForward --gpus 0
run scalar-acc genai_ce --full-graphs repros/models/genai/static/CrossEntropyForward --gpus 0
# 5. genai SoftmaxForward full graph
run base       genai_sm --full-graphs repros/models/genai/static/SoftmaxForward --gpus 0
run scalar-acc genai_sm --full-graphs repros/models/genai/static/SoftmaxForward --gpus 0
echo "ALL DONE $(date +%T)"
