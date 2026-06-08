# sum_sum_75ee5fc5ea19

## Status

- Family: `structured_pool_upsample_backward_reduce`
- Closure status: `at_floor`
- Artifact: `repros/canonical/sum_sum_75ee5fc5ea19/oracle_electra_ce_backward_full.py`
- Classification: `BANDWIDTH_BOUND`

## Full-Scope Contract

The oracle computes the complete Electra causal-LM cross-entropy-backward
fragment returned by `Repro.forward`, including shifted-label ignore-index
handling, scalar f32 division, f32 `exp(logit - row_shift0 - row_shift1)`,
residual add, the right-padded `[32768, 30524]` materialized output, the
transposed `[30522, 32768]` view, and the `[30522]` column sum, while replacing
the dense equality-built one-hot tensor and its row reduction with an equivalent
guarded label scalar.

- Inputs: `T([64, 513], i64)`, `T([64, 512, 30522], f32)` x2, `T([32768, 1], f32)` x2, scalars
- Models: hf_ElectraForCausalLM_train (2 variants)
- Correctness: PASS, output0_max_diff=0.00e+00, output1_max_diff=0.00e+00, output2_max_diff=1.56e-02

## Timings

- Oracle: 7018.88 us
- torch.compile (combo+CDT): 6915.07 us
- Ratio: 0.985x (compile is faster; oracle is not a valid floor)

## Gap Diagnosis

The full-scope CUDAGraph measurement shows Inductor already at floor because the
required dense logits/residual reads, two dense output materializations, natural
f32 exp work, and exact f32 column reduction dominate. Inductor cannot materially
improve this isolated repro through scheduler fusion, scatter-reduce, split-K,
algebraic elimination, or recompute fusion without removing required outputs.
Classification: BANDWIDTH_BOUND -- record as at floor, while a guarded one-hot
reduction rewrite may still simplify related graphs where the dense materialized
outputs are not live.

## Validation

```bash
python repros/canonical/sum_sum_75ee5fc5ea19/oracle_electra_ce_backward_full.py --check
python repros/canonical/sum_sum_75ee5fc5ea19/oracle_electra_ce_backward_full.py --bench --warmup 10 --rep 50
```
