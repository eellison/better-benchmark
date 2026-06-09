# Softmax Codegen Gap Analysis: Inductor vs Oracle

## Summary

The Inductor-generated softmax kernel for bf16[8192, 262144] is **15-145% slower** than the
oracle depending on Triton compilation non-determinism. The gap has two components:

1. **`libdevice.exp` in the normalization pass**: 15-142% penalty (bimodal)
2. **Codegen structural overhead** (2D layout, int64 indexing): 3.7-4.3% penalty

The claimed "2286 us / 15% slower" figure matches the `libdevice.exp` "fast compilation"
mode. The worst case is 2.4x slower when Triton generates suboptimal PTX for `libdevice.exp`.

## Measured Performance (Blackwell B200, SM100, R0_BLOCK=8192)

| Variant | Time (us) | vs Oracle |
|---------|-----------|-----------|
| Oracle (BLOCK_N=8192) | 1951 | 1.00x |
| Inductor, tl_math.exp (use_fast_math=True) | 2036 | 1.043x |
| Inductor, libdevice.exp (best case) | 2280-2440 | 1.17-1.25x |
| Inductor, libdevice.exp (worst case) | 4690-4790 | 2.40-2.45x |

Note: `libdevice.exp` performance is bimodal due to Triton compiler non-determinism.
The kernel compiles identically in source but produces different PTX quality across runs.

## Root Cause: `libdevice.exp` vs `tl_math.exp`

Inductor uses two different `exp` implementations within the same kernel:

- **Pass 1 (online softmax reduction):** Uses `tl_math.exp` because the scalar
  accumulator codegen path (`prepare_softmax_online`) hardcodes it via `triton_helpers.exp()`
  with `use_fast_math` passed through.

- **Pass 2 (normalization: `exp(x - max) / sum`):** Uses `libdevice.exp` because
  `aten.exp.default` is lowered via `TritonOverrides.exp()` which checks
  `config.use_fast_math` (False by default).

The code path in `torch/_inductor/codegen/triton.py` line 1466-1469:
```python
if config.use_fast_math:
    return f"tl_math.exp({x})"
else:
    return f"libdevice.exp({x})"
```

On Blackwell SM100, `libdevice.exp` (IEEE754 precise, using `__nv_expf` -> special function
unit) is catastrophically slower than `tl_math.exp` (fast math `ex2.approx` instruction)
for this bandwidth-bound workload. The extra precision has no value in softmax normalization
because the result is immediately truncated to bf16.

## Secondary Gap: Codegen Overhead (3.7-4.3%)

With `use_fast_math=True`, the remaining gap comes from:

### 1. 2D tensor layout [XBLOCK=1, R0_BLOCK] vs 1D [BLOCK_N] (~3.7%)

Oracle:
```python
cols = block_start + tl.arange(0, BLOCK_N)        # 1D vector
m_new = tl.maximum(m_i, tl.max(x, axis=0))        # reduce axis=0 on 1D
```

Inductor:
```python
r0_base = tl.arange(0, R0_BLOCK)[None, :].to(tl.int64)  # 2D [1, BLOCK]
_tmp3_max_block_max = triton_helpers.max2(tmp2, 1)        # reduce axis=1 on 2D
_tmp3_max_new = triton_helpers.maximum(_tmp3_max, _tmp3_max_block_max)
```

The 2D layout causes Triton to generate slightly different PTX (extra dimension tracking,
different register allocation) even though XBLOCK=1 makes it logically equivalent to 1D.

### 2. int64 indexing (~0-1%)

Inductor:
```python
xoffset = tl.program_id(0).to(tl.int64) * XBLOCK
xindex = xoffset + tl.arange(0, XBLOCK)[:, None].to(tl.int64)
r0_base = tl.arange(0, R0_BLOCK)[None, :].to(tl.int64)
```

Oracle:
```python
row_idx = tl.program_id(0)          # int32
cols = block_start + tl.arange(0, BLOCK_N)   # int32
```

The int64 indexing is used because `xnumel` and `r0_numel` are declared as `i64` in the
signature (for dynamic shapes support). With static shapes [8192, 262144], int32 suffices.

### 3. Eviction policy in pass 2 (~0.6%)

Inductor uses `eviction_policy='evict_first'` for loads in the normalization pass (pass 2).
The oracle has no eviction hint (uses default). Since data is only read once in pass 2,
`evict_first` is actually the correct hint - the slight cost is likely from interacting
with the L2 cache state left by pass 1.

### 4. Negligible differences (measured at 0%)

- `tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])` when XBLOCK=1: no-op, compiler eliminates
- `tl.where(_tmp3_max_new == float("-inf"), 1.0, exp(...))`: branch eliminated at compile time after first iteration
- Dead variables (`r0_mask`, `roffset`, `rindex`, `rbase`, `xmask`): DCE'd by compiler
- `.to(tl.float32)` on already-f32 value: no-op
- bf16 store vs f32 store: same (output pointer type determines actual store width)

## Detailed Variant Benchmark (BLOCK_SIZE=8192, consistent run)

| Variant | Time (us) | vs Oracle |
|---------|-----------|-----------|
| Oracle (1D, masks, tl.exp, bf16 store) | 1951 | 1.000x |
| Oracle (1D, NO masks) | 1950 | 0.999x |
| Oracle logic, 2D layout [1,BLOCK] | 2021 | 1.036x |
| Inductor EXACT (libdevice.exp, worst case) | 4736 | 2.427x |
| Inductor + tl_math.exp | 2036 | 1.043x |
| Inductor no broadcast_to + tl_math.exp | 2036 | 1.043x |
| Inductor no broadcast + evict_last in pass2 | 2024 | 1.037x |
| Inductor no broadcast + bf16 store | 2035 | 1.043x |
| Inductor no -inf check + bf16 store | 2034 | 1.042x |
| Inductor oracle-like (int32, no dead code) | 2023 | 1.037x |

The 3.7% floor comes from the 2D [1, BLOCK] tensor shape causing different PTX generation.
This matches the "Oracle logic, 2D layout" variant (also 3.6%).

## Generated Kernel Code (Inductor, use_fast_math=False)

```python
@triton.jit
def triton_red_fused_convert_element_type_div_exp_prepare_softmax_online_sub_0(
    in_ptr0, out_ptr2, xnumel, r0_numel, XBLOCK: tl.constexpr, R0_BLOCK: tl.constexpr
):
    xnumel = 8192
    r0_numel = 262144
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0).to(tl.int64) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None].to(tl.int64)
    r0_base = tl.arange(0, R0_BLOCK)[None, :].to(tl.int64)
    x0 = xindex
    _tmp3_max = tl.full([XBLOCK], float('-inf'), tl.float32)
    _tmp3_sum = tl.full([XBLOCK], 0.0, tl.float32)

    # Pass 1: Online softmax (running max + sum_exp)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 262144*x0), None, eviction_policy='evict_last').to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
        _tmp3_max_block_max = triton_helpers.max2(tmp2, 1)
        _tmp3_max_new = triton_helpers.maximum(_tmp3_max, _tmp3_max_block_max)
        _tmp3_sum_correction = tl.where(
            _tmp3_max_new == float("-inf"), 1.0,
            tl_math.exp((_tmp3_max - _tmp3_max_new).to(tl.float32))  # <-- tl_math.exp (fast)
        )
        _tmp3_sum_block = tl.sum(
            tl_math.exp((tmp2 - _tmp3_max_new[:, None]).to(tl.float32)), 1  # <-- tl_math.exp (fast)
        )
        _tmp3_sum = _tmp3_sum * _tmp3_sum_correction + _tmp3_sum_block
        _tmp3_max = _tmp3_max_new

    tmp3 = _tmp3_max[:, None]
    tmp4 = _tmp3_sum[:, None]

    # Pass 2: Normalize
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_1 = r0_index
        tmp5 = tl.load(in_ptr0 + (r0_1 + 262144*x0), None, eviction_policy='evict_first').to(tl.float32)
        tmp6 = tmp5.to(tl.float32)
        tmp7 = tmp6 - tmp3
        tmp8 = libdevice.exp(tmp7)  # <-- libdevice.exp (SLOW - 2.4x total kernel slowdown!)
        tmp9 = (tmp8 / tmp4)
        tmp10 = tmp9.to(tl.float32)
        tl.store(out_ptr2 + (r0_1 + 262144*x0), tmp10, None)
```

## Oracle Kernel Code

```python
@triton.jit
def online_softmax_fwd_kernel(input_ptr, output_ptr, N: tl.constexpr, BLOCK_N: tl.constexpr):
    row_idx = tl.program_id(0)
    row_start = row_idx * N

    m_i = float("-inf")
    l_i = 0.0

    # Pass 1: Online max + sum_exp
    for block_start in tl.range(0, N, BLOCK_N):
        cols = block_start + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(input_ptr + row_start + cols, mask=mask, other=float("-inf")).to(tl.float32)
        m_new = tl.maximum(m_i, tl.max(x, axis=0))
        l_i = l_i * tl.exp(m_i - m_new) + tl.sum(tl.exp(x - m_new), axis=0)  # tl.exp (fast)
        m_i = m_new

    # Pass 2: Normalize
    for block_start in tl.range(0, N, BLOCK_N):
        cols = block_start + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(input_ptr + row_start + cols, mask=mask, other=float("-inf")).to(tl.float32)
        out = tl.exp(x - m_i) / l_i  # tl.exp (fast)
        tl.store(output_ptr + row_start + cols, out.to(tl.bfloat16), mask=mask)
```

## Autotuner Config

The autotuner picks XBLOCK=1, R0_BLOCK=16384, num_warps=8 for this kernel.
The oracle uses BLOCK_N=8192 (fixed). With `use_fast_math=True`, the autotuner's
choice of R0_BLOCK=16384 vs oracle's 8192 matters less (both give ~2050 us).
Without fast math, the libdevice.exp penalty dominates.

## Recommendations

1. **Primary fix: Use `tl_math.exp` for softmax normalization** (137% improvement)
   - In `TritonOverrides.exp()`, use fast math for softmax patterns where the result
     will be cast to lower precision (bf16/fp16).
   - Or: make `use_fast_math=True` the default (it already is for the online softmax
     reduction pass, creating an inconsistency).
   - Risk: slightly different numerical results for edge cases near exp overflow/underflow.
     For softmax normalization (where input is `x - max`, always <= 0), this is safe.

2. **Secondary fix: 1D codegen for XBLOCK=1** (3.7% improvement)
   - When the autotuner selects XBLOCK=1, generate 1D code instead of 2D.
   - Remove `[:, None]` and `[None, :]` reshaping, use `axis=0` reductions.
   - This requires a codegen path change in `TritonKernel`.

3. **Tertiary: int32 indexing for static shapes** (<1% improvement)
   - When shapes are statically known to fit in int32, avoid `.to(tl.int64)`.

## Files to Change

| File | Change |
|------|--------|
| `torch/_inductor/codegen/triton.py` L1466 | Change `exp()` to use `tl_math.exp` for softmax-related ops (or default to fast math) |
| `torch/_inductor/config.py` L265 | Consider making `use_fast_math = True` the default |
| `torch/_inductor/codegen/triton.py` | Specialized 1D codegen path when XBLOCK=1 |
