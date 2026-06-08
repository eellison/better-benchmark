# sum_sum_2d5b06e4c2a8

## Status

- Family: `bandwidth_bound_channel_reduction`
- Closure status: `bad_oracle`
- Artifact: `repros/canonical/sum_sum_2d5b06e4c2a8/oracle_algebraic_channel_reduction.py`
- Classification: `BAD_ORACLE`

## Full-Scope Contract

The oracle consumes the same inputs as `repro.py` and returns a single
`float32[72]` channel reduction output. It implements a MobileNetV3
BN-affine/ReLU gated-gradient channel vector via atomic accumulation.

## Timings

- Oracle: 50.94 us
- torch.compile (default): 44.8 us
- Ratio: 0.879x (compiler is faster)

## Gap Diagnosis

The compiled code is already faster than the oracle (ratio 0.879x). Inductor
already reaches the same CUDAGraph-measured floor for these two generic
reductions. The oracle atomic-accumulation approach does not beat Inductor's
strategy for this shape (C=72, spatial=large). Classification: BANDWIDTH_BOUND --
runtime is dominated by mandatory f32 scan of the two NCHW inputs plus
affine/ReLU math. No action needed.

## Validation

- `oracle_algebraic_channel_reduction.py --check`: PASS
- `oracle_algebraic_channel_reduction.py --bench`: ratio 0.879x, status BAD_ORACLE
