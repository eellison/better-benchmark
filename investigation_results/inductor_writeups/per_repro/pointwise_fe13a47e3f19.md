# pointwise_fe13a47e3f19


## Measured Timings
- Oracle: 26.34 us
- Compile (CDT): 30.50 us
- Ratio: 1.16x

## Classification: REAL_TO_COMPLEX_CAST_MATERIALIZATION

## Current Result

- Family: `complex_cast_view`
- Oracle path: `repros/canonical/pointwise_fe13a47e3f19/oracle_complex_cast_view.py`
- Correctness: PASS (assumed from oracle docstring)
- Status: `real_gap`

## Diagnosis

The oracle computes the complete view plus float32-to-complex64 cast as one direct Triton materialization, streaming the contiguous `[16384, 768]` input storage and writing each `complex64[32, 512, 768]` element with one packed 64-bit real-plus-zero-imaginary store, whereas Inductor lowers the isolated metadata view and dtype conversion through generic complex-cast pointwise code. Inductor's pointwise codegen has no real-to-complex64 cast pattern that preserves the virtual view while emitting packed complex stores.

## Root cause

Inductor's complex dtype handling decomposes complex64 into separate real/imaginary operations, potentially doubling the store traffic or using sub-optimal store patterns. The oracle packs each float32 value with a zero imaginary component into a single uint64 store, achieving 1 load + 1 store per element at 8 bytes out per element (matching the complex64 output size).

## Kernel count

- Oracle: 1 kernel (_complex_cast_kernel with autotune over BLOCK_N)
- Inductor: 1 kernel (generic complex-cast pointwise, potentially less efficient stores)

## Config exploration

Standard configs (cd=True, combo_kernels, multi_kernel) are unlikely to help since this is a codegen quality issue in the complex dtype path, not a fusion or scheduling issue.

## Recommendation

Add a guarded real-to-complex materialization lowering in Inductor that packs the exact float32 real bits with a zero imaginary lane in one output store when the source dtype is float32 and the target is complex64.

## Relevant files

- `/tmp/pytorch-work/torch/_inductor/codegen/triton.py` (complex store codegen)
- `/tmp/pytorch-work/torch/_inductor/lowering.py` (dtype conversion lowering)
- `/tmp/pytorch-work/torch/_inductor/ir.py` (complex layout handling)
