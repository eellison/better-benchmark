# sum_sum_sum_4de1559f24a7 (MegatronBert LN-backward)


## Measured Timings
- Oracle: measurement failed (runtime error)
- Compile (CDT): 61.25 us
- Ratio: N/A

## Classification: COOPERATIVE_SPLIT_K

## Measurements (fresh cache, all fixes including combo_kernels)

| Config | Time (us) | Kernels |
|--------|-----------|---------|
| compile (cd+combo+scalar_acc) | 72.5 | 3 triton + 2 ATen workspace reductions |
| oracle (cooperative split-K) | 53.2 | 2 (1 main + 1 tiny finalizer) |
| gap ratio | 1.36x | |

## Graph Structure

```
mm_274 + mm_276 + mm_278 [8192,1024] -> x (three matmul grad views summed)
x * arg11_1 [1024] -> weighted
weighted -> sum([2]) -> row_sum [16,512,1]  (row reduction)
weighted * arg216_1 -> sum([2]) -> row_dot [16,512,1]  (row reduction)
layernorm_backward(weighted, row_sum, row_dot, arg641_1) + add_137 -> add_tensor_2
add_tensor_2 * arg215_1(mask) * 1.111 -> mul_tensor_7 [16,512,1024]
x * arg216_1 -> sum([0,1]) -> sum_dim_int_list_2 [1024]  (column reduction)
x -> sum([0,1]) -> sum_dim_int_list_3 [1024]  (column reduction)
mul_tensor_7 -> view -> sum([0]) -> view_default_4 [1024]  (column reduction)
mul_tensor_7 -> view -> permute([1,0]) -> [1024,8192]
```

## What Inductor Does

Same pattern as Longformer (sum_sum_sum_1d10e6314ef6):
1. **Main kernel** (mix-order-reduction): row-persistent processing of all 8192
   rows. Computes row reductions, LN-backward, residual add, dropout mask,
   writes `mul_tensor_7` (32MB). Accumulates workspace partials for the two
   column reductions of `x` and `x*arg216_1`.
2. **Workspace finalization** (ATen): reduces [1024,1024] partials to [1024].
3. **Outer reduction** (2 triton kernels): re-reads the materialized
   `mul_tensor_7` [16,512,1024] = 32MB to compute `view_default_4` [1024].

## What the Oracle Does Differently

Single-pass streaming kernel that:
- Computes row-local LN-backward including 3 input adds, affine weight, row
  reductions, residual add, and dropout masking
- Writes the output to memory (required for the permute/transpose return)
- Accumulates **three** column partials (x*rhs, x, out) as a side effect

Small finalizer sums the partials. Total: 1 pass over inputs, no re-read.

## Root Cause

Identical to the Longformer case but with larger data (32MB vs 24MB output
buffer), making the re-read penalty proportionally larger (19.3us vs 10.4us).

The scheduler cannot represent "write output AND accumulate output into column
partial" in one kernel. The `sum_dim_int_list_4` reduction of `mul_tensor_7`
is scheduled as a separate consumer, re-reading the entire 32MB buffer that
was just written.

## Inductor Change Needed

Same fix as Longformer: extend the mix-order-reduction or persistent-reduction
codegen to allow a "side accumulator" that records column partials of the
values being stored. When the output of a producer kernel feeds a compatible
column reduction, the producer should accumulate partials into workspace and
emit a small finalizer.

For this specific case: the main kernel already writes workspace for 2 column
sums. Adding a 3rd workspace channel for `sum(mul_tensor_7, [0,1])` would
eliminate the 2 outer-reduction kernel launches and the 32MB re-read.

## Recommendation

Status: COOPERATIVE_SPLIT_K / needs_work. The 19.3us gap (1.36x) is the
largest of the three and clearly addressable. This is the same fix as the
Longformer case -- implementing it once covers both.

Priority: medium-high (32MB re-read elimination, affects MegatronBert and
similar large-hidden-dim transformer LN-backward patterns).
