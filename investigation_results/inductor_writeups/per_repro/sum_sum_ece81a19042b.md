# sum_sum_ece81a19042b

## Status

- Family: `multi_output_reduction_templates`
- Closure status: `at_floor`
- Artifact: `repros/canonical/sum_sum_ece81a19042b/oracle_ghostnet_dual_sum.py`
- Classification: `BANDWIDTH_BOUND`

## Full-Scope Contract

The oracle computes the complete GhostNet BN-backward-style captured scope,
including the sliced-add producer, both sibling per-channel reductions, the
dependent dense epilogue, and the 8-element scale-gradient side output, while
sharing one split-K Triton reduction with two f32 accumulators and recomputing
the cheap sliced-add/sub producers for the final tensor epilogue.

- Inputs: `T([512, 16, 112, 112], f32)`, `T([512, 8, 112, 112], f32)` x2, `T([1, 8, 1, 1], f32)`, `T([8], f32)` x2
- Models: timm_ghostnet_100_train (3 variants), timm_mobilevit_s_train (3 variants), torchbench_timm_vovnet
- Correctness: PASS, output0_max_diff=1.91e-06, output1_max_diff=2.93e-03

## Timings

- Oracle: 394.43 us
- torch.compile (combo+CDT): 402.75 us
- Ratio: 1.021x (effectively at floor)

## Gap Diagnosis

Inductor already lands within 2.1% of this full-scope floor, so the possible
scheduler savings from co-planning the sibling reductions are hidden by the
required full-tensor reads and write. The mandatory bandwidth dominates, not
launch structure. Classification: BANDWIDTH_BOUND -- no action needed.

## Validation

```bash
python repros/canonical/sum_sum_ece81a19042b/oracle_ghostnet_dual_sum.py --check
python repros/canonical/sum_sum_ece81a19042b/oracle_ghostnet_dual_sum.py --bench --warmup 10 --rep 50
```
