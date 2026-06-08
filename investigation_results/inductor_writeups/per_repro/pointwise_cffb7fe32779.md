# pointwise_cffb7fe32779

## Classification: METADATA_ONLY_ALIAS_CONSTRUCTION

## Current Result

- Oracle path: `repros/canonical/pointwise_cffb7fe32779/oracle_qkv_view_alias.py`
- Model: hf_DistillGPT2_infer_000
- Correctness: PASS
- Oracle: 2.53 us
- Compile (cd=True, combo=True): 2.69 us
- Ratio: 1.063
- Status: GOOD (marginal, noise-level gap)

## Config Exploration

| Config | Compile (us) | Ratio | Status |
|--------|-------------|-------|--------|
| Default (cd=True, combo=True) | 2.69 | 1.063 | GOOD |
| multi_kernel=2 | 2.82 | 1.023 | AT_FLOOR |
| multi_kernel=3 | 2.78 | 0.978 | AT_FLOOR |
| use_fast_math=True | 2.69 | 1.012 | AT_FLOOR |

The gap closes with any config change, confirming it is measurement noise at the device floor.

## Root Cause Analysis

The repro is a pure metadata operation: `view -> split(768, dim=2) -> view -> permute([0,2,1,3])` on a `[16384, 2304]` tensor to produce three `[32, 12, 512, 64]` views (Q, K, V heads for DistilGPT-2). The oracle implements this as three `as_strided` calls at different storage offsets - no GPU kernels are launched.

The CUDA Graph capture warning ("The CUDA Graph is empty") confirms there is no actual GPU work here. Both oracle and compile produce zero-kernel metadata-only results. The ~0.16us "gap" is pure measurement noise from CUDA graph capture overhead on an empty graph.

## Kernel Count

- Inductor: 0 kernels (metadata only, no GPU launch)
- Oracle: 0 kernels (as_strided alias construction)

## Conclusion

This is not a meaningful performance gap. The entire computation is host-side tensor metadata manipulation (view/split/permute are all stride-only ops). The 1.063 ratio is within measurement noise at the 2.5us device floor, as confirmed by the gap disappearing with different configs. No code changes needed.
