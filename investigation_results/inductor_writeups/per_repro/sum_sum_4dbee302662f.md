# sum_sum_4dbee302662f

## Status

- Family: `bandwidth_bound_channel_reduction`
- Closure status: `bad_oracle`
- Artifact: `repros/canonical/sum_sum_4dbee302662f/oracle_fused_channel_reduction.py`
- Classification: `BAD_ORACLE`

## Full-Scope Contract

The oracle consumes the same inputs as `repro.py` and returns a single
`float32[224]` channel reduction output. It implements the complete RegNet
BN-affine/ReLU gated-gradient channel vector with spatial f32 reduction,
sigmoid-derivative epilogue, and batch/channel f32 reduction.

## Timings

- Oracle: 37.66 us
- torch.compile (default): 34.5 us
- Ratio: 0.916x (compiler is faster)

## Gap Diagnosis

The compiled code is already faster than the oracle (ratio 0.916x). Tuned
Inductor already reaches the same CUDAGraph-measured floor for these two
generic reductions. The dominant work is the mandatory f32 scan of the two
`[32, 224, 56, 56]` inputs plus affine/ReLU math, while the `[32, 224]`
intermediate and second reduction are tiny. Classification: BANDWIDTH_BOUND --
no action needed.

## Validation

- `oracle_fused_channel_reduction.py --check`: PASS
- `oracle_fused_channel_reduction.py --bench`: ratio 0.916x, status BAD_ORACLE
