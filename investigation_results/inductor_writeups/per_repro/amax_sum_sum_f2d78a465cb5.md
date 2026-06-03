# amax_sum_sum_f2d78a465cb5

## Queue Position

- Rank: new
- Family: `online_softmax_cross_entropy`
- Owner: unassigned
- Closure status: `needs_inductor_gap_closure`
- Oracle status: `true_oracle_measured`

## Current Gap

- Best compile (perf_counter): `46.426 us`
- Oracle (CUDA events): `30.560 us`
- Compile/oracle gap: `1.52x`
- Oracle path: `repros/canonical/amax_sum_sum_f2d78a465cb5/oracle_shifted_causal_lm_cross_entropy_mean.py`
- Correctness: PASS
- Model: hf_GPTJForCausalLM_infer_000
- Shape: logits=f32[128, 50400], tokens=i64[1, 128]

## Oracle Description

Triton online softmax cross-entropy kernel matching the inference variant of
GPT-J causal LM loss. Same structure as amax_sum_sum_846668f0b88f: single-pass
online row logsumexp, shifted target gather, masked loss/count accumulation,
scalar mean. The inference variant has slightly different op ordering (ne_scalar
before the logits reshape) but identical algorithmic structure.

## Diagnosis

Classification: NEW_PATTERN

Nearly identical to amax_sum_sum_846668f0b88f. The graph decomposes the same
shifted-label cross-entropy mean into separate generic softmax, gather, masked
sum, count, and divide reductions. The oracle collapses all into a single
online row pass. Inductor lacks the pattern for this fused form.

## Inductor Closure Path

- Same implementation track as amax_sum_sum_846668f0b88f: the same cross-entropy
  fusion pattern would close both repros simultaneously.
- Gating and benchmark policy identical to sibling.

## Done Criteria

- Canonical oracle measured: YES (30.560 us, correct)
- Gap actionable: YES (1.52x, same NEW_PATTERN as sibling)
