# PR perf verification plan (2026-07-15)
Arms: PYTHONPATH shadow worktrees under /tmp/pt-verify-wt/{base,rsqrt,ce-hoist,online-softmax,scalar-acc}
Fresh TORCHINDUCTOR_CACHE_DIR per arm under /tmp/scratch_space/pr_perf_verify/cache_<arm>.
All runs: scripts/bench_parallel.py, --no-cd, clock lock default on, sequential invocations.

Sets:
- SET_RSQRT: pointwise_2c331ef4f17f (23 sh), pointwise_9128d8745e42 (3), mean_3840584eef9a (2) --all-shapes
  (walk a73d1583b34: 2.0-2.6x on these; var_mean_* named in task already have aten.rsqrt -> unaffected, ~1.0x in walk)
- SET_OS: amax_sum_02064a1e60ac (1), amax_sum_sum_04ddf882ff17 (1), amax_sum_sum_3f000d9caa57 (3) --all-shapes
- SET_CE: amax_sum_sum_42988b64e7f9 (1), amax_sum_sum_4bf8a79efec4 (2) --all-shapes (walk ab29345: 2.76x/2.15x/2.10x)
- GENAI_CE: repros/models/genai/static/CrossEntropyForward --full-graphs
- GENAI_SM: repros/models/genai/static/SoftmaxForward --full-graphs

Runs: base×{SET_RSQRT,SET_OS,SET_CE,GENAI_CE,GENAI_SM}; rsqrt×SET_RSQRT; online-softmax×SET_OS;
ce-hoist×{SET_CE,GENAI_CE}; scalar-acc×{GENAI_CE,GENAI_SM}
