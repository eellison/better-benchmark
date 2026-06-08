# sum_sum_e106b4fbe172

## Status

- Family: `multi_output_reduction_templates`
- Closure status: `good`
- Artifact: `repros/canonical/sum_sum_e106b4fbe172/oracle_mt5_dropout_dual_reduction.py`
- Classification: `SCHEDULER_FUSION`

## Full-Scope Contract

The oracle computes the complete MT5 dropout/backward scope from Repro.forward,
sharing the 17-term f32 add/dropout producer across the sibling `[512]` channel
reduction, the per-row reduction, and the dependent transposed dense epilogue.

- Inputs: 15 `T([4096, 512], f32)` tensors, `T([32, 128, 512], f32/b8)`, `T([512], f32)`, `T([32, 128, 1], f32)`, shapes
- Models: hf_MT5ForConditionalGeneration_train (2 variants)
- Correctness: PASS, output0_max_diff=2.44e-04, output1_max_diff=7.63e-06

## Timings

- Oracle: 67.90 us
- torch.compile (combo+CDT): 90.08 us
- Ratio: 1.327x (oracle is 32.7% faster -- valid gap)

## Gap Diagnosis

Inductor currently schedules the same full graph with extra separation around
the sibling reductions and the layout-changing epilogue, so it cannot keep the
shared producer and reduction partials co-planned across both outputs.

The fix is SCHEDULER_FUSION: teach the scheduler/codegen to emit a multi-output
reduction template that fuses a shared same-numel producer into sibling
reductions plus the row-finalized permuted epilogue while preserving the captured
f32 operation order.

## Inductor Closure Path

- Implementation track: multi-output reduction with transposed epilogue fusion.
- Candidate hook: recognize that the 17-term add/dropout producer feeds both a channel reduction and a row reduction with a permuted output, enabling single-pass multi-accumulator codegen.

## Validation

```bash
python repros/canonical/sum_sum_e106b4fbe172/oracle_mt5_dropout_dual_reduction.py --check
python repros/canonical/sum_sum_e106b4fbe172/oracle_mt5_dropout_dual_reduction.py --bench --warmup 10 --rep 50
```
