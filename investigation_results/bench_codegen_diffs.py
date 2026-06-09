"""
Isolate specific code quality differences between Inductor-generated softmax
and the oracle. Each variant changes one aspect to measure its contribution.
"""
import torch
import triton
import triton.language as tl
import sys
sys.path.insert(0, '/tmp/scratch_space/better_benchmark')

from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math

BLOCK_SIZE = 8192
M = 8192
N = 262144


# === Variant 0: Oracle (baseline, fastest) ===
@triton.jit
def oracle_kernel(input_ptr, output_ptr, N: tl.constexpr, BLOCK_N: tl.constexpr):
    """Oracle: 1D vectors, scalar accumulators, tl.exp, no mask (N divisible by BLOCK_N)."""
    row_idx = tl.program_id(0)
    row_start = row_idx * N

    m_i = float("-inf")
    l_i = 0.0

    for block_start in tl.range(0, N, BLOCK_N):
        cols = block_start + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(input_ptr + row_start + cols, mask=mask, other=float("-inf")).to(tl.float32)
        m_new = tl.maximum(m_i, tl.max(x, axis=0))
        l_i = l_i * tl.exp(m_i - m_new) + tl.sum(tl.exp(x - m_new), axis=0)
        m_i = m_new

    for block_start in tl.range(0, N, BLOCK_N):
        cols = block_start + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(input_ptr + row_start + cols, mask=mask, other=float("-inf")).to(tl.float32)
        out = tl.exp(x - m_i) / l_i
        tl.store(output_ptr + row_start + cols, out.to(tl.bfloat16), mask=mask)


# === Variant 1: Oracle without masks (N=262144 is divisible by 8192) ===
@triton.jit
def oracle_nomask_kernel(input_ptr, output_ptr, N: tl.constexpr, BLOCK_N: tl.constexpr):
    """Oracle with no mask - since 262144 % 8192 == 0, mask is always True."""
    row_idx = tl.program_id(0)
    row_start = row_idx * N

    m_i = float("-inf")
    l_i = 0.0

    for block_start in tl.range(0, N, BLOCK_N):
        cols = block_start + tl.arange(0, BLOCK_N)
        x = tl.load(input_ptr + row_start + cols).to(tl.float32)
        m_new = tl.maximum(m_i, tl.max(x, axis=0))
        l_i = l_i * tl.exp(m_i - m_new) + tl.sum(tl.exp(x - m_new), axis=0)
        m_i = m_new

    for block_start in tl.range(0, N, BLOCK_N):
        cols = block_start + tl.arange(0, BLOCK_N)
        x = tl.load(input_ptr + row_start + cols).to(tl.float32)
        out = tl.exp(x - m_i) / l_i
        tl.store(output_ptr + row_start + cols, out.to(tl.bfloat16))


# === Variant 2: Inductor-style 2D layout [1, BLOCK_N] but oracle logic ===
@triton.jit
def oracle_2d_layout_kernel(input_ptr, output_ptr, N: tl.constexpr, BLOCK_N: tl.constexpr):
    """Oracle logic but using [XBLOCK=1, R0_BLOCK] 2D tensor layout like Inductor."""
    XBLOCK: tl.constexpr = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    r0_base = tl.arange(0, BLOCK_N)[None, :]
    x0 = xindex

    _m_i = tl.full([XBLOCK], float('-inf'), tl.float32)
    _l_i = tl.full([XBLOCK], 0.0, tl.float32)

    for r0_offset in tl.range(0, N, BLOCK_N):
        r0_index = r0_offset + r0_base
        r0_1 = r0_index
        x = tl.load(input_ptr + (r0_1 + N*x0), None).to(tl.float32)
        m_new = triton_helpers.maximum(_m_i, triton_helpers.max2(x, 1))
        _l_i = _l_i * tl_math.exp((_m_i - m_new).to(tl.float32)) + tl.sum(tl_math.exp((x - m_new[:, None]).to(tl.float32)), 1)
        _m_i = m_new

    m_final = _m_i[:, None]  # [1, 1]
    l_final = _l_i[:, None]  # [1, 1]

    for r0_offset in tl.range(0, N, BLOCK_N):
        r0_index = r0_offset + r0_base
        r0_1 = r0_index
        x = tl.load(input_ptr + (r0_1 + N*x0), None).to(tl.float32)
        out = tl_math.exp((x - m_final).to(tl.float32)) / l_final
        tl.store(output_ptr + (r0_1 + N*x0), out.to(tl.bfloat16), None)


# === Variant 3: Inductor exact code with int64 indexing ===
@triton.jit
def inductor_exact_kernel(in_ptr0, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    """Exact copy of Inductor generated code."""
    xnumel = 8192
    r0_numel = 262144
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0).to(tl.int64) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None].to(tl.int64)
    xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
    r0_base = tl.arange(0, R0_BLOCK)[None, :].to(tl.int64)
    rbase = r0_base
    x0 = xindex
    _tmp3_max = tl.full([XBLOCK], float('-inf'), tl.float32)
    _tmp3_sum = tl.full([XBLOCK], 0.0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = tl.full([R0_BLOCK], True, tl.int1)[None, :]
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 262144*x0), None, eviction_policy='evict_last').to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])

        _tmp3_max_block_max = triton_helpers.max2(tmp2, 1)
        _tmp3_max_new = triton_helpers.maximum(_tmp3_max, _tmp3_max_block_max)
        _tmp3_sum_correction = tl.where(
            _tmp3_max_new == float("-inf"),
            1.0,
            tl_math.exp((_tmp3_max - _tmp3_max_new).to(tl.float32))
        )
        _tmp3_sum_block = tl.sum(
            tl_math.exp((tmp2 - _tmp3_max_new[:, None]).to(tl.float32)),
            1
        )
        _tmp3_sum = _tmp3_sum * _tmp3_sum_correction + _tmp3_sum_block
        _tmp3_max = _tmp3_max_new

    tmp3 = _tmp3_max[:, None]
    tmp4 = _tmp3_sum[:, None]
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = tl.full([R0_BLOCK], True, tl.int1)[None, :]
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp5 = tl.load(in_ptr0 + (r0_1 + 262144*x0), None, eviction_policy='evict_first').to(tl.float32)
        tmp6 = tmp5.to(tl.float32)
        tmp7 = tmp6 - tmp3
        tmp8 = libdevice.exp(tmp7)
        tmp9 = (tmp8 / tmp4)
        tmp10 = tmp9.to(tl.float32)
        tl.store(out_ptr2 + (r0_1 + 262144*x0), tmp10, None)


# === Variant 4: Inductor but with tl.exp instead of libdevice.exp ===
@triton.jit
def inductor_tlexp_kernel(in_ptr0, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    """Inductor code but using tl_math.exp everywhere (pass 2 uses tl.exp-like)."""
    xnumel = 8192
    r0_numel = 262144
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0).to(tl.int64) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None].to(tl.int64)
    xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
    r0_base = tl.arange(0, R0_BLOCK)[None, :].to(tl.int64)
    rbase = r0_base
    x0 = xindex
    _tmp3_max = tl.full([XBLOCK], float('-inf'), tl.float32)
    _tmp3_sum = tl.full([XBLOCK], 0.0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = tl.full([R0_BLOCK], True, tl.int1)[None, :]
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 262144*x0), None, eviction_policy='evict_last').to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])

        _tmp3_max_block_max = triton_helpers.max2(tmp2, 1)
        _tmp3_max_new = triton_helpers.maximum(_tmp3_max, _tmp3_max_block_max)
        _tmp3_sum_correction = tl.where(
            _tmp3_max_new == float("-inf"),
            1.0,
            tl_math.exp((_tmp3_max - _tmp3_max_new).to(tl.float32))
        )
        _tmp3_sum_block = tl.sum(
            tl_math.exp((tmp2 - _tmp3_max_new[:, None]).to(tl.float32)),
            1
        )
        _tmp3_sum = _tmp3_sum * _tmp3_sum_correction + _tmp3_sum_block
        _tmp3_max = _tmp3_max_new

    tmp3 = _tmp3_max[:, None]
    tmp4 = _tmp3_sum[:, None]
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = tl.full([R0_BLOCK], True, tl.int1)[None, :]
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp5 = tl.load(in_ptr0 + (r0_1 + 262144*x0), None, eviction_policy='evict_first').to(tl.float32)
        tmp6 = tmp5.to(tl.float32)
        tmp7 = tmp6 - tmp3
        tmp8 = tl_math.exp(tmp7)  # <-- changed from libdevice.exp
        tmp9 = (tmp8 / tmp4)
        tmp10 = tmp9.to(tl.float32)
        tl.store(out_ptr2 + (r0_1 + 262144*x0), tmp10, None)


# === Variant 5: Inductor without the redundant broadcast_to ===
@triton.jit
def inductor_no_broadcast_kernel(in_ptr0, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    """Inductor code without broadcast_to (since XBLOCK=1, data is already [1, R0_BLOCK])."""
    xnumel = 8192
    r0_numel = 262144
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0).to(tl.int64) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None].to(tl.int64)
    r0_base = tl.arange(0, R0_BLOCK)[None, :].to(tl.int64)
    x0 = xindex
    _tmp3_max = tl.full([XBLOCK], float('-inf'), tl.float32)
    _tmp3_sum = tl.full([XBLOCK], 0.0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 262144*x0), None, eviction_policy='evict_last').to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        # Skip broadcast_to - tmp1 is already [1, R0_BLOCK]

        _tmp3_max_block_max = triton_helpers.max2(tmp1, 1)
        _tmp3_max_new = triton_helpers.maximum(_tmp3_max, _tmp3_max_block_max)
        _tmp3_sum_correction = tl.where(
            _tmp3_max_new == float("-inf"),
            1.0,
            tl_math.exp((_tmp3_max - _tmp3_max_new).to(tl.float32))
        )
        _tmp3_sum_block = tl.sum(
            tl_math.exp((tmp1 - _tmp3_max_new[:, None]).to(tl.float32)),
            1
        )
        _tmp3_sum = _tmp3_sum * _tmp3_sum_correction + _tmp3_sum_block
        _tmp3_max = _tmp3_max_new

    tmp3 = _tmp3_max[:, None]
    tmp4 = _tmp3_sum[:, None]
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_1 = r0_index
        tmp5 = tl.load(in_ptr0 + (r0_1 + 262144*x0), None, eviction_policy='evict_first').to(tl.float32)
        tmp6 = tmp5.to(tl.float32)
        tmp7 = tmp6 - tmp3
        tmp8 = tl_math.exp(tmp7)
        tmp9 = (tmp8 / tmp4)
        tmp10 = tmp9.to(tl.float32)
        tl.store(out_ptr2 + (r0_1 + 262144*x0), tmp10, None)


# === Variant 6: Inductor but with evict_last in pass 2 (like oracle) ===
@triton.jit
def inductor_evict_last_kernel(in_ptr0, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    """Same as inductor_no_broadcast but pass 2 uses no eviction policy (default)."""
    xnumel = 8192
    r0_numel = 262144
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0).to(tl.int64) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None].to(tl.int64)
    r0_base = tl.arange(0, R0_BLOCK)[None, :].to(tl.int64)
    x0 = xindex
    _tmp3_max = tl.full([XBLOCK], float('-inf'), tl.float32)
    _tmp3_sum = tl.full([XBLOCK], 0.0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 262144*x0), None, eviction_policy='evict_last').to(tl.float32)
        tmp1 = tmp0.to(tl.float32)

        _tmp3_max_block_max = triton_helpers.max2(tmp1, 1)
        _tmp3_max_new = triton_helpers.maximum(_tmp3_max, _tmp3_max_block_max)
        _tmp3_sum_correction = tl.where(
            _tmp3_max_new == float("-inf"),
            1.0,
            tl_math.exp((_tmp3_max - _tmp3_max_new).to(tl.float32))
        )
        _tmp3_sum_block = tl.sum(
            tl_math.exp((tmp1 - _tmp3_max_new[:, None]).to(tl.float32)),
            1
        )
        _tmp3_sum = _tmp3_sum * _tmp3_sum_correction + _tmp3_sum_block
        _tmp3_max = _tmp3_max_new

    tmp3 = _tmp3_max[:, None]
    tmp4 = _tmp3_sum[:, None]
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_1 = r0_index
        # Use evict_last like oracle (instead of evict_first)
        tmp5 = tl.load(in_ptr0 + (r0_1 + 262144*x0), None, eviction_policy='evict_last').to(tl.float32)
        tmp6 = tmp5.to(tl.float32)
        tmp7 = tmp6 - tmp3
        tmp8 = tl_math.exp(tmp7)
        tmp9 = (tmp8 / tmp4)
        tmp10 = tmp9.to(tl.float32)
        tl.store(out_ptr2 + (r0_1 + 262144*x0), tmp10, None)


# === Variant 7: Inductor but output in bf16 (like oracle) instead of f32 store ===
@triton.jit
def inductor_bf16_store_kernel(in_ptr0, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    """Inductor code but stores bf16 directly (oracle does .to(tl.bfloat16) before store)."""
    xnumel = 8192
    r0_numel = 262144
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0).to(tl.int64) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None].to(tl.int64)
    r0_base = tl.arange(0, R0_BLOCK)[None, :].to(tl.int64)
    x0 = xindex
    _tmp3_max = tl.full([XBLOCK], float('-inf'), tl.float32)
    _tmp3_sum = tl.full([XBLOCK], 0.0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 262144*x0), None, eviction_policy='evict_last').to(tl.float32)
        tmp1 = tmp0.to(tl.float32)

        _tmp3_max_block_max = triton_helpers.max2(tmp1, 1)
        _tmp3_max_new = triton_helpers.maximum(_tmp3_max, _tmp3_max_block_max)
        _tmp3_sum_correction = tl.where(
            _tmp3_max_new == float("-inf"),
            1.0,
            tl_math.exp((_tmp3_max - _tmp3_max_new).to(tl.float32))
        )
        _tmp3_sum_block = tl.sum(
            tl_math.exp((tmp1 - _tmp3_max_new[:, None]).to(tl.float32)),
            1
        )
        _tmp3_sum = _tmp3_sum * _tmp3_sum_correction + _tmp3_sum_block
        _tmp3_max = _tmp3_max_new

    tmp3 = _tmp3_max[:, None]
    tmp4 = _tmp3_sum[:, None]
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_1 = r0_index
        tmp5 = tl.load(in_ptr0 + (r0_1 + 262144*x0), None, eviction_policy='evict_first').to(tl.float32)
        tmp6 = tmp5.to(tl.float32)
        tmp7 = tmp6 - tmp3
        tmp8 = tl_math.exp(tmp7)
        tmp9 = (tmp8 / tmp4)
        tmp10 = tmp9.to(tl.bfloat16)  # <-- store as bf16 like oracle
        tl.store(out_ptr2 + (r0_1 + 262144*x0), tmp10, None)


# === Variant 8: Inductor with _tmp3_sum_correction without the == -inf check ===
@triton.jit
def inductor_no_inf_check_kernel(in_ptr0, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    """Inductor code but correction uses exp directly (no -inf check). Like oracle."""
    xnumel = 8192
    r0_numel = 262144
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0).to(tl.int64) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None].to(tl.int64)
    r0_base = tl.arange(0, R0_BLOCK)[None, :].to(tl.int64)
    x0 = xindex
    _tmp3_max = tl.full([XBLOCK], float('-inf'), tl.float32)
    _tmp3_sum = tl.full([XBLOCK], 0.0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 262144*x0), None, eviction_policy='evict_last').to(tl.float32)
        tmp1 = tmp0.to(tl.float32)

        _tmp3_max_block_max = triton_helpers.max2(tmp1, 1)
        _tmp3_max_new = triton_helpers.maximum(_tmp3_max, _tmp3_max_block_max)
        # Simplified correction: just use exp(old_max - new_max) directly
        # (when max is -inf, exp(-inf - new) = 0, which is correct since sum is also 0)
        _tmp3_sum_correction = tl_math.exp((_tmp3_max - _tmp3_max_new).to(tl.float32))
        _tmp3_sum_block = tl.sum(
            tl_math.exp((tmp1 - _tmp3_max_new[:, None]).to(tl.float32)),
            1
        )
        _tmp3_sum = _tmp3_sum * _tmp3_sum_correction + _tmp3_sum_block
        _tmp3_max = _tmp3_max_new

    tmp3 = _tmp3_max[:, None]
    tmp4 = _tmp3_sum[:, None]
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_1 = r0_index
        tmp5 = tl.load(in_ptr0 + (r0_1 + 262144*x0), None, eviction_policy='evict_first').to(tl.float32)
        tmp6 = tmp5.to(tl.float32)
        tmp7 = tmp6 - tmp3
        tmp8 = tl_math.exp(tmp7)
        tmp9 = (tmp8 / tmp4)
        tmp10 = tmp9.to(tl.bfloat16)
        tl.store(out_ptr2 + (r0_1 + 262144*x0), tmp10, None)


# === Variant 9: Closest to oracle - int32 indexing, no deadcode, bf16 store ===
@triton.jit
def inductor_oracle_like_kernel(in_ptr0, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    """Inductor-style structure but optimized to match oracle as closely as possible."""
    xnumel = 8192
    r0_numel = 262144
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    x0 = xindex
    _tmp3_max = tl.full([XBLOCK], float('-inf'), tl.float32)
    _tmp3_sum = tl.full([XBLOCK], 0.0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        tmp0 = tl.load(in_ptr0 + (r0_index + 262144*x0), None).to(tl.float32)

        _tmp3_max_block_max = triton_helpers.max2(tmp0, 1)
        _tmp3_max_new = triton_helpers.maximum(_tmp3_max, _tmp3_max_block_max)
        _tmp3_sum_correction = tl_math.exp((_tmp3_max - _tmp3_max_new).to(tl.float32))
        _tmp3_sum_block = tl.sum(
            tl_math.exp((tmp0 - _tmp3_max_new[:, None]).to(tl.float32)),
            1
        )
        _tmp3_sum = _tmp3_sum * _tmp3_sum_correction + _tmp3_sum_block
        _tmp3_max = _tmp3_max_new

    tmp3 = _tmp3_max[:, None]
    tmp4 = _tmp3_sum[:, None]
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        tmp5 = tl.load(in_ptr0 + (r0_index + 262144*x0), None).to(tl.float32)
        tmp7 = tmp5 - tmp3
        tmp8 = tl_math.exp(tmp7)
        tmp9 = (tmp8 / tmp4)
        tl.store(out_ptr2 + (r0_index + 262144*x0), tmp9.to(tl.bfloat16), None)


def benchmark_cuda_graph(fn, warmup=5, rep=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(rep):
        start_event.record()
        graph.replay()
        end_event.record()
        torch.cuda.synchronize()
        times.append(start_event.elapsed_time(end_event))
    times.sort()
    return times[len(times)//2] * 1000.0  # us


def main():
    torch.manual_seed(42)
    x = torch.randn(M, N, dtype=torch.bfloat16, device='cuda')
    out = torch.empty_like(x)

    grid = (M,)

    print("=" * 70)
    print("Isolating Code Quality Differences: Inductor vs Oracle Softmax")
    print(f"Shape: bf16[{M}, {N}], BLOCK_SIZE={BLOCK_SIZE}")
    print("=" * 70)

    variants = [
        ("Oracle (1D, masks, tl.exp, bf16 store)",
         lambda: oracle_kernel[grid](x, out, N=N, BLOCK_N=BLOCK_SIZE)),
        ("Oracle (1D, NO masks)",
         lambda: oracle_nomask_kernel[grid](x, out, N=N, BLOCK_N=BLOCK_SIZE)),
        ("Oracle logic, 2D layout [1,BLOCK]",
         lambda: oracle_2d_layout_kernel[grid](x, out, N=N, BLOCK_N=BLOCK_SIZE)),
        ("Inductor EXACT (int64, broadcast, libdevice.exp, f32 store)",
         lambda: inductor_exact_kernel[grid](x, out, 8192, 262144, XBLOCK=1, R0_BLOCK=BLOCK_SIZE)),
        ("Inductor + tl_math.exp (no libdevice)",
         lambda: inductor_tlexp_kernel[grid](x, out, 8192, 262144, XBLOCK=1, R0_BLOCK=BLOCK_SIZE)),
        ("Inductor no broadcast_to + tl_math.exp",
         lambda: inductor_no_broadcast_kernel[grid](x, out, 8192, 262144, XBLOCK=1, R0_BLOCK=BLOCK_SIZE)),
        ("Inductor no broadcast + evict_last in pass2",
         lambda: inductor_evict_last_kernel[grid](x, out, 8192, 262144, XBLOCK=1, R0_BLOCK=BLOCK_SIZE)),
        ("Inductor no broadcast + bf16 store",
         lambda: inductor_bf16_store_kernel[grid](x, out, 8192, 262144, XBLOCK=1, R0_BLOCK=BLOCK_SIZE)),
        ("Inductor no -inf check + bf16 store",
         lambda: inductor_no_inf_check_kernel[grid](x, out, 8192, 262144, XBLOCK=1, R0_BLOCK=BLOCK_SIZE)),
        ("Inductor oracle-like (int32, no dead code, bf16 store)",
         lambda: inductor_oracle_like_kernel[grid](x, out, 8192, 262144, XBLOCK=1, R0_BLOCK=BLOCK_SIZE)),
    ]

    results = []
    for name, fn in variants:
        us = benchmark_cuda_graph(fn, warmup=5, rep=20)
        results.append((name, us))
        print(f"  {us:7.1f} us  {name}")

    oracle_us = results[0][1]
    print(f"\n--- Relative to Oracle ---")
    for name, us in results:
        ratio = us / oracle_us
        delta_pct = (ratio - 1) * 100
        print(f"  {ratio:.3f}x ({delta_pct:+.1f}%)  {name}")


if __name__ == "__main__":
    with torch.no_grad():
        main()
