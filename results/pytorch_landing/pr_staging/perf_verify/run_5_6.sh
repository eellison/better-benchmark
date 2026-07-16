#!/bin/bash
# Sequential A/B perf verification for PR5 (U10) + PR6 (U30).
# One bench_parallel invocation per (arm, set). Arms differ ONLY in
# PYTHONPATH (shadow worktree) + TORCHINDUCTOR_CACHE_DIR.
set -u
cd /tmp/scratch_space/better_benchmark
OUT=/tmp/scratch_space/pr56_verify/perf
WT=/tmp/pt-verify-wt
mkdir -p "$OUT"

run() {
  local arm="$1"; shift
  local tag="$1"; shift
  local outfile="$OUT/${tag}__${arm}.json"
  if [ -s "$outfile" ]; then echo "SKIP $tag $arm (exists)"; return 0; fi
  echo "=== RUN $tag arm=$arm $(date +%T) ==="
  PYTHONPATH=$WT/$arm TORCHINDUCTOR_CACHE_DIR=$OUT/cache_${tag}_${arm} \
    python scripts/bench_parallel.py "$@" --no-cd --output "$outfile" \
    > "$OUT/log_${tag}__${arm}.txt" 2>&1
  echo "=== DONE $tag arm=$arm rc=$? $(date +%T) ==="
}

SET_U10=(
  repros/canonical/pointwise_00475df23925 repros/canonical/pointwise_3c17f0120a71
  repros/canonical/pointwise_41944c71d2d8 repros/canonical/pointwise_666951cd0da5
  repros/canonical/pointwise_7308e0024674 repros/canonical/pointwise_99e028e77568
  repros/canonical/pointwise_ba25986618ef repros/canonical/var_mean_33d24326eae6
  repros/canonical/var_mean_b12103db1177 repros/canonical/var_mean_e67fddfba074
  repros/canonical/var_mean_fd789e584775 repros/canonical/var_mean_var_mean_74b57fcc4507
  repros/canonical/var_mean_var_mean_92c45aff3580 repros/canonical/var_mean_var_mean_e642e2ea37a3
)
SET_U30=(
  repros/canonical/sum_sum_sum_260a107eaf32 repros/canonical/sum_sum_sum_11d45d703ba6
  repros/canonical/sum_sum_sum_2261b2f5694a repros/canonical/sum_sum_sum_0703a79bc871
  repros/canonical/sum_sum_sum_907cbb3d9f19 repros/canonical/sum_sum_sum_ecce309d13e3
)

# PR5 U10 (all 14 affected)
run base           u10_set "${SET_U10[@]}" --all-shapes --gpus 0,1,2,3
run layout-sinking u10_set "${SET_U10[@]}" --all-shapes --gpus 0,1,2,3
# PR6 U30 (6 of 52 affected)
run base         u30_set "${SET_U30[@]}" --all-shapes --gpus 0,1,2,3
run mor-finalize u30_set "${SET_U30[@]}" --all-shapes --gpus 0,1,2,3
# repeat branch arms for within-session noise check
run layout-sinking u10_set2 "${SET_U10[@]}" --all-shapes --gpus 0,1,2,3
run mor-finalize   u30_set2 "${SET_U30[@]}" --all-shapes --gpus 0,1,2,3
echo "ALL DONE $(date +%T)"
