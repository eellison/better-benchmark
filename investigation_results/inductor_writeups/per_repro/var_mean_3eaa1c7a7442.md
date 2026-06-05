# var_mean_3eaa1c7a7442

## Compile: 7.55us, Oracle: 7.10us, Gap: 1.063x

## Diagnosis: BANDWIDTH_BOUND

## Root cause: The oracle computes the Reformer residual-add LayerNorm in one shape-specialized Triton row kernel: addmm [4096,256] f16 -> view [1,4096,256] -> add(residual) -> f32 var_mean([2], correction=0, keepdim=True) -> LayerNorm(eps=1e-12) -> f16 cast -> two views aliasing one [1,4096,256] output. Inductor already targets this exact same fused normalization envelope. The remaining 6.3% gap is within measurement noise for this small, bandwidth-dominated workload.

The workload processes 4096 rows of hidden=256 (next_pow2=256, fits perfectly in one tile). Both Inductor and the oracle read ~4MB of input data and write ~2MB of output. At this scale, launch overhead and cache effects dominate over algorithmic differences.

## Kernel count
- Inductor: 1 kernel (fused add+layernorm+cast+view)
- Oracle: 1 kernel (same scope)

## Config exploration results
- multi_kernel=0: 43.02us, multi_kernel=1: 46.67us, multi_kernel=2: 45.63us, multi_kernel=3: 44.39us
- No config helps; multi_kernel makes it worse
- coord_descent_tuning already enabled

## Fix path: This is a near-floor case. The 6.3% gap is within the noise range for a bandwidth-bound kernel at this scale. The two output views alias the same backing tensor, so there's no extra work. No structural fix is needed.

## Status: near_floor

## Details
- Model: torchbench_hf_Reformer_infer_005
- Pattern: addmm [4096,256] f16 -> view [1,4096,256] -> add(residual [1,4096,256] f16) -> f32 cast -> var_mean([2], correction=0, keepdim=True) -> LayerNorm(eps=1e-12, affine) -> f16 cast -> two views [4096,256] (aliased output)
- Shapes: 4096 rows, hidden=256, f16 input/output, f32 internal
- Two returned views are aliases of one [1,4096,256] tensor -- no extra compute
- Gap is 6.3%, within noise floor for this bandwidth-bound kernel size
