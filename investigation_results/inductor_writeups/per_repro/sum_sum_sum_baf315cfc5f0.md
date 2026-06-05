# sum_sum_sum_baf315cfc5f0

## Compile: FAILS (NameError), Oracle: N/A (correctness passes), Gap: N/A

## Diagnosis: CODEGEN_BUG_DEAD_BUFFER_ELIMINATION

## Root cause

Inductor has a codegen bug where the scheduler incorrectly eliminates a buffer (`buf6`) that is still needed as a return value. The generated wrapper code references `buf6` in the return statement but never allocates it.

The repro computes a GPT-Neo layer-norm-backward pattern:
- Three matmul inputs are summed, multiplied by gamma, row-reduced (sum, dot)
- The layer-norm backward gradient is computed
- The result is added to a residual, then:
  - Returned as a `[2048, 4096]` permuted view (output 2)
  - Reduced to `[2048]` (output 3)
  - Two column reductions produce `[2048]` vectors (outputs 0, 1)

The bug: The `view_default_3` node (which views `add_tensor_2` as `[4096, 2048]`) is used by both `permute_default` (returned as output 2) and `sum_dim_int_list_4` (reduced to output 3). The scheduler correctly computes the sum but incorrectly eliminates the full `[4096, 2048]` buffer because it believes the only remaining consumer (the permute) can be served by a view/alias. However, the buffer it tries to alias (`buf6`) was never allocated.

Generated code shows:
- `buf7 = empty_strided_cuda((1, 2048, 32), ...)` -- partial sums for column reduction
- The return references `reinterpret_tensor(buf6, (2048, 4096), (1, 2048), 0)` -- buf6 never defined

The actual `add_181 + mul_tensor_4` result (the full `[32, 128, 2048]` tensor) is needed for the permuted output but was never stored to a named buffer.

## Config exploration

| Config | Result | Notes |
|--------|--------|-------|
| default (combo_kernels=True, cdt=True) | NameError: buf6 | Bug |
| combo_kernels=False, multi_kernel=0 | NameError: buf6 | Same bug |
| multi_kernel=1 | NameError: buf6 | Same bug |
| multi_kernel=2 | NameError: buf6 | Same bug |

The bug reproduces across all configurations tested.

## Kernel count (from generated code before crash)
- Inductor: 3 kernels generated (persistent per-row reduce, looped column reduce, per-column finalize)
- Oracle: 2 kernels (fused row-store + partial column reduce, column finalize)

## Status: CODEGEN_BUG (blocks benchmarking)

This is a genuine Inductor codegen bug on the `pr-184905` branch at commit `c465f1751c6`. The scheduler's buffer elimination pass incorrectly removes a buffer needed for a permuted view output when that same buffer's data is also consumed by a reduction.

## Fix direction

The bug is likely in the scheduler's reuse/elimination logic in `/tmp/pytorch-work/torch/_inductor/scheduler.py` or the output buffer planning in `/tmp/pytorch-work/torch/_inductor/codegen/triton.py`. The condition should check that a buffer used as a return-value view/permute cannot be eliminated even if its other consumers (reductions) don't need it materialized.

## File references
- Oracle: repros/canonical/sum_sum_sum_baf315cfc5f0/oracle_cooperative_split_k.py
- Model: hf_GPTNeoForCausalLM_train_001
- Pattern: Layer-norm backward with permuted view output + column reductions
- Generated (broken): /tmp/torchinductor_dev/tv/ctvrkwyuoj6w5tdf7xxovqjydpeczi6vz7rkkfs6vpa4wnhapmcz.py
- Bug location: /tmp/pytorch-work/torch/_inductor/scheduler.py (buffer elimination/reuse)
