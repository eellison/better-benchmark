
## MEGA-COMMIT DECOMPOSITION (deferred — eventually)
97385fb3273 measured at +1.95pp in the per-commit walk, but it's a GRAB-BAG of ~6 features
(BN-split-threshold, MOR Triton finalize, CE gather-into-softmax, reduction_chaining [dead code],
evict_first/scalar_acc, + core simd/triton/ir hooks). The commit-walk CANNOT split this (it's one
commit) — needs the SELECTIVE-REVERT / flag-toggle approach: from HEAD (or a state with the
mega-commit applied), revert/flag-off each internal feature one at a time, measure the model-delta.
Most of those features ARE flag-gated (A0 found the flags), so it's flag-ablation WITHIN the commit.
PRIORITY: LOWER — +1.95pp is modest vs rsqrt +5.64pp; harder mechanism; do after the headline
verification + A2. The dead reduction_chaining.py should just be dropped (never imported).
