# sum_sum_2eb4fbbc04e1

## Status

- Family: `bandwidth_bound_bn_backward`
- Closure status: `at_floor`
- Artifact: `repros/canonical/sum_sum_2eb4fbbc04e1/oracle_visformer_bn_backward.py`
- Classification: `AT_FLOOR`

## Full-Scope Contract

The oracle consumes the same inputs as `repro.py` and returns two outputs:
`float32[192]` channel vector and `float32[128, 192, 28, 28]` full tensor
epilogue. It implements the complete Visformer BN-backward-style scope with
shared `sum(x * grad)` and `sum(x)` channel reductions, one fused f32 summary
finalizer, and the full tensor epilogue.

## Timings

- Oracle: 81.57 us
- torch.compile (default): 81.6 us
- Ratio: 1.0x (effectively tied)

## Gap Diagnosis

The oracle and compiled code are at the same performance floor (ratio 1.0x).
Inductor already emits the same split channel-reduction structure and f32
epilogue. Classification: BANDWIDTH_BOUND -- runtime is dominated by the
required two activation/gradient reads, add-input read, and full output store.
No action needed.

## Validation

- `oracle_visformer_bn_backward.py --check`: PASS
- `oracle_visformer_bn_backward.py --bench`: ratio 1.0x, status AT_FLOOR
