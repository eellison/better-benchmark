# pointwise_b51064ab890f — QKV Layout View

## Summary
- **Model**: hf_GPTJForCausalLM_infer_000
- **Classification**: AT_FLOOR
- **Ratio**: 0.981x (oracle 6.78us vs compile 6.66us)
- **Status**: Inductor already at or below oracle performance

## Analysis

The repro performs `permute([0,2,1,3]) -> clone(contiguous) -> view` on a [1, 16, 128, 256] f32 tensor. This is a pure layout transformation (transpose + copy). Inductor's compiled kernel (6.66us) already matches or beats the oracle (6.78us).

This is a memory-bandwidth-limited copy operation where Inductor's generic codegen is already optimal.

## No Action Required

Inductor is already at or below oracle performance for this repro.
