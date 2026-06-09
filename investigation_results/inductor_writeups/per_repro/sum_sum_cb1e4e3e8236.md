# sum_sum_cb1e4e3e8236

## Status

- Family: `multi_output_reduction_templates`
- Closure status: `good`
- Artifact: `repros/canonical/sum_sum_cb1e4e3e8236/oracle_megatron_ce_column_sum.py`
- Classification: `ALGEBRAIC_ELIMINATION`

## Full-Scope Contract

The oracle computes the complete MegatronBERT causal-LM cross-entropy-backward
fragment returned by `Repro.forward`, including shifted-label ignore-index
handling, scalar f32 division, f32 `exp(logit - row_shift0 - row_shift1)`,
residual add, the returned `[29056, 8192]` permute view, and the returned
`[29056]` column sum, while replacing the materialized dense equality-built
one-hot tensor and its row reduction with the equivalent guarded label scalar.

- Inputs: `T([16, 513], i64)`, `T([16, 512, 29056], f32)` x2, `T([8192, 1], f32)` x2, shapes
- Models: hf_MegatronBertForCausalLM_train (2 variants)
- Correctness: PASS, output0_max_diff=0.00e+00, output1_max_diff=4.69e-02

## Timings

- Oracle: 856.70 us
- torch.compile (combo+CDT): 1094.75 us
- Ratio: 1.278x (oracle is 27.8% faster -- valid gap)

## Gap Diagnosis

Inductor currently lowers the decomposed one-hot `eq/where/mul/sum`, exponential
epilogue, residual add, permute metadata, and column sum as generic dense
pointwise/reduction work over the whole vocabulary. Algebraic simplification does
not canonicalize one-hot masked reductions into per-row guarded label formulas
and then share that producer with the sibling layout-changing materialization.

The fix is ALGEBRAIC_ELIMINATION: add a guarded one-hot reduction rewrite before
reduction scheduling and let multi-output codegen fuse the dense epilogue store
with the compatible column reduction.

## Inductor Closure Path

- Implementation track: algebraic one-hot reduction elimination.
- Candidate hook: recognize `eq(labels.unsqueeze(-1), iota) * grad` followed by `sum(dim=-1)` as equivalent to a guarded label scalar, then share the simplified producer with the dense materialization and sibling column sum.
- Related repro: sum_sum_75ee5fc5ea19 (Electra variant, same pattern but at floor due to different shapes).

## Validation

```bash
python repros/canonical/sum_sum_cb1e4e3e8236/oracle_megatron_ce_column_sum.py --check
python repros/canonical/sum_sum_cb1e4e3e8236/oracle_megatron_ce_column_sum.py --bench --warmup 10 --rep 50
```
