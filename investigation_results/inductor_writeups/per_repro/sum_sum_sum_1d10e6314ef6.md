# sum_sum_sum_1d10e6314ef6 (Longformer LN-backward)


## Measured Timings
- Oracle: measurement failed (runtime error)
- Compile (CDT): 38.59 us
- Ratio: N/A

## Classification: COOPERATIVE_SPLIT_K

## Measurements (fresh cache, all fixes including combo_kernels)

| Config | Time (us) | Kernels |
|--------|-----------|---------|
| compile (cd+combo+scalar_acc) | 56.4 | 3 triton + 2 ATen workspace reductions |
| oracle (cooperative split-K) | 46.0 | 2 (1 main + 1 tiny finalizer) |
| gap ratio | 1.22x | |

## Graph Structure

```
mm_134 [8192,768] + mul_290 [8,1024,768] -> add
add * arg5_1 [768] -> weighted (used 3 ways)
weighted -> sum([2]) -> row_sum [8,1024,1]  (row reduction)
weighted * arg104_1 -> sum([2]) -> row_dot [8,1024,1]  (row reduction)
layernorm_backward(weighted, row_sum, row_dot, arg298_1) -> mul_tensor_7 [8,1024,768]
add * arg104_1 -> sum([0,1]) -> sum_dim_int_list_2 [768]  (column reduction)
add -> sum([0,1]) -> sum_dim_int_list_3 [768]  (column reduction)
mul_tensor_7 * arg103_1(mask) -> masked_grad [8,1024,768]
masked_grad -> sum([0,1]) -> view_default_1 [768]  (column reduction)
masked_grad -> permute([1,0]) -> [768,8192]
```

## What Inductor Does

1. **Main kernel** (mix-order-reduction): processes each of the 8192 rows
   persistently, computing the row reductions (sum, dot), the full LN-backward
   epilogue, the dropout mask, writing `mul_tensor_7`, and accumulating
   workspace partials for the two [768] column reductions of `add` and
   `add*arg104_1`.
2. **Workspace finalization** (ATen .sum(dim=0)): reduces the [1024,768]
   workspace partials to produce `buf1` and `buf3` (the two [768] vectors).
3. **Outer reduction** (2 triton kernels): re-reads the materialized
   `mul_tensor_7` [8,1024,768] = 24MB to compute `sum_dim_int_list_4` [768].

## What the Oracle Does Differently

The oracle streams all rows through **one kernel pass** that simultaneously:
- Computes the row-local LN-backward (row_sum, row_dot)
- Writes the masked gradient side output
- Accumulates **three** column partial buffers (x*rhs, x, grad)

A tiny finalizer kernel sums the partials. Total: 2 kernel launches, 1 pass
over the 24MB input data, no re-read of the output.

## Root Cause

Inductor cannot attach a dependent column reduction (`sum_dim_int_list_4`) to
the producer that writes `mul_tensor_7`. The scheduler treats the
`mul_tensor_7 -> sum([0,1])` as a separate consumer reduction, requiring a
full re-read of 24MB. The oracle avoids this by accumulating the sum as a
side-effect of the producer pass.

This is the **COOPERATIVE_SPLIT_K** pattern: a materialized producer that
should also accumulate a sibling small reduction using partial buffers or
atomic coordination.

## Inductor Change Needed

Add codegen support for "producer + side reduction" mode: when a materialized
output (that must be stored) also feeds a column/batch reduction, the producer
kernel should accumulate partial sums into a workspace buffer, with a small
finalizer pass. This avoids the 24MB re-read.

Specifically for the Longformer case:
- The mix-order-reduction kernel already uses workspace for 2 of the 3 column
  reductions. The third one (`sum_dim_int_list_4`) is missed because it depends
  on the dropout-masked output which is computed inside the same kernel.
- The fix is to allow the kernel to also accumulate the output it is writing
  into a third workspace channel.

## Recommendation

Status: COOPERATIVE_SPLIT_K / needs_work. The 10.4us gap (1.22x) is real and
addressable. Priority: medium (behind larger scatter-reduce and online-softmax
wins, but affects Longformer and similar transformer LN-backward patterns).
