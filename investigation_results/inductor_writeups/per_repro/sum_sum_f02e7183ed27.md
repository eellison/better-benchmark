# sum_sum_f02e7183ed27

## Status

- Family: `unclassified_priority_gap`
- Closure status: `at_floor`
- Artifact: `repros/canonical/sum_sum_f02e7183ed27/oracle_nfnet_silu_channel_sum.py`
- Classification: `BANDWIDTH_BOUND`

## Full-Scope Contract

The oracle computes the complete NFNet SiLU-gradient reduction returned by
Repro.forward, fusing the captured `exp(-x) + 1` reciprocal chain, residual add,
0.2/2.0 scaling, spatial f32 sum, exact sigmoid-derivative gate, and final
channel sum with one per-(N,C) Triton reduction plus a channel finalizer.

- Inputs: `T([128, 1536, 14, 14], f32)` x4, `T([128, 1536, 1, 1], f32)`
- Models: timm_nfnet_l0_train (4 variants)
- Correctness: PASS, max_diff=1.14e-05

## Timings

- Oracle: 178.18 us
- torch.compile (combo+CDT): 182.37 us
- Ratio: 1.024x (effectively at floor)

## Gap Diagnosis

Tuned Inductor already lowers the same full scope near the
memory/transcendental floor. The dominant work is the mandatory scan of four
dense f32 activation tensors plus natural exp math and channel reduction.
Classification: BANDWIDTH_BOUND -- at floor unless broader
libdevice-exp/reduction codegen improvements move both implementations.

## Validation

```bash
python repros/canonical/sum_sum_f02e7183ed27/oracle_nfnet_silu_channel_sum.py --check
python repros/canonical/sum_sum_f02e7183ed27/oracle_nfnet_silu_channel_sum.py --bench --warmup 10 --rep 50
```
