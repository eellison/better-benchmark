# amax_sum_sum_846668f0b88f

## Queue Position

- Rank: new
- Family: `online_softmax_cross_entropy`
- Owner: unassigned
- Closure status: `closed_by_inline_recomputable_producers`
- Oracle status: `true_oracle_measured`

## Current Gap

- Best compile: `18.24 us`
- Oracle: `25.25 us`
- Compile/oracle gap: `0.72x` (compile beats oracle, no gap remains)
- Previous gap: `1.66x` (48.44us compile vs 29.18us oracle)
- Oracle path: `repros/canonical/amax_sum_sum_846668f0b88f/oracle_shifted_ignore_index_cross_entropy_mean.py`
- Correctness: PASS
- Model: hf_GPTJForCausalLM_train_000
- Shape: logits=f32[128, 50400], tokens=i64[1, 128]

## Fix: inline_recomputable_producers (f58d2545cd2)

The extension to `inline_recomputable_producers` that handles cheap non-broadcast
producers closed this gap completely. Compile now beats the oracle (0.72x ratio).
Re-measured 2026-06-08.

## Oracle Description

Triton online softmax cross-entropy kernel that processes each row once with
scalar online max and sum-exp accumulators, loads the shifted target logit,
writes per-row loss and valid count, then reduces to a scalar mean in a second
tiny kernel. No intermediate log-softmax buffer materialized.

## Diagnosis

Classification: NEW_PATTERN

The repro decomposes shifted-label ignore-index cross entropy into
pad/slice/view/amax/sub/exp/sum/log/gather/where/sum/div, requiring Inductor
to materialize and re-read the full f32[128, 50400] log-softmax intermediate
across separate reduction and pointwise kernels. The oracle fuses all of this
into a single-pass online row accumulator with negligible scratch (only 128
per-row scalars). Inductor has no pattern-match or template for this combined
shifted-target + masked-mean cross-entropy idiom.

## Inductor Closure Path

- Implementation track: Add an Inductor cross-entropy-mean pattern that detects
  shifted log_softmax + gather + masked sum/count/div and emits an online
  row-accumulator Triton template plus a tiny scalar mean epilogue.
- Gating: gate on (a) vocab size > 1024, (b) presence of ignore-index mask,
  (c) shifted-label pad pattern in the graph.
- Benchmark policy: compare default compile vs online-softmax template; accept
  only if strictly faster and correct within fp32 tolerance.

## Done Criteria

- Canonical oracle measured: YES (29.184 us, correct)
- Gap closed: YES — compile 18.24us beats oracle 25.25us after inline_recomputable_producers fix (f58d2545cd2)
