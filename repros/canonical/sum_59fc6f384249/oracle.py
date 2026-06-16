"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle materializes the returned bf16 QKV matrix and transpose alias while computing the returned q/v reduction slices directly from the original strided inputs with the captured f32-sum then bf16-round then f32-cast boundary, whereas Inductor first writes the full cloned QKV buffer then schedules reduction work that rereads it and computes the unreturned middle k columns; Inductor cannot do this today because its scheduler does not fuse a returned layout materialization with a reduction consumer or push final slices into the reduction domain; the fix is SCHEDULER_FUSION: allow mixed materialization-plus-reduction scheduling with slice-aware column pruning."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


ROWS = 25216
COLS = 2304
HEAD_COLS = 768
Q_OR_V_COLS = 1536
TOKENS = 197
BATCH_STRIDE = 151296
TOKEN_STRIDE = 768
PARTIAL_CHUNKS = 59
PARTIAL_ROWS = 428


@triton.jit
def _round_to_bf16_f32(x):
    return tl.inline_asm_elementwise(
        "{ .reg .b16 t; cvt.rn.bf16.f32 t, $1; cvt.f32.bf16 $0, t; }",
        "=f,f",
        [x],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _materialize_k_kernel(
    k_ptr,
    qkv_ptr,
    total: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < total
    head_col = offsets % 768
    row = offsets // 768
    batch = row // 197
    token = row - batch * 197
    src_offset = batch * 151296 + token * 768 + head_col
    value = tl.load(k_ptr + src_offset, mask=mask, other=0.0)
    tl.store(qkv_ptr + row * 2304 + 768 + head_col, value, mask=mask)


@triton.jit
def _materialize_qv_partial_sum_kernel(
    q_ptr,
    v_ptr,
    qkv_ptr,
    partial_ptr,
    BLOCK_COL: tl.constexpr,
    RBLOCK: tl.constexpr,
):
    chunk = tl.program_id(0)
    col_base = tl.program_id(1) * BLOCK_COL
    cols = col_base + tl.arange(0, BLOCK_COL)[:, None]
    out_cols = tl.where(cols < 768, cols, cols + 768)
    r_base = tl.arange(0, RBLOCK)[None, :]
    acc = tl.full([BLOCK_COL, RBLOCK], 0.0, tl.float32)

    for r_offset in tl.range(0, 428, RBLOCK):
        r = r_offset + r_base
        rows = chunk * 428 + r
        valid = (cols < 1536) & (r < 428) & (rows < 25216)

        head_col = cols % 768
        batch = rows // 197
        token = rows - batch * 197
        src_offset = batch * 151296 + token * 768 + head_col

        q_val = tl.load(
            q_ptr + src_offset,
            mask=valid & (cols < 768),
            eviction_policy="evict_first",
            other=0.0,
        )
        v_val = tl.load(
            v_ptr + src_offset,
            mask=valid & (cols >= 768),
            eviction_policy="evict_first",
            other=0.0,
        )
        values = tl.where(cols < 768, q_val, v_val)
        tl.store(qkv_ptr + rows * 2304 + out_cols, values, mask=valid)
        acc += values

    sums = tl.sum(acc, axis=1)[:, None]
    tl.store(partial_ptr + chunk * 1536 + cols, sums, mask=cols < 1536)


@triton.jit
def _final_qv_sum_kernel(
    partial_ptr,
    sum_ptr,
    BLOCK_COL: tl.constexpr,
    RBLOCK: tl.constexpr,
):
    col_base = tl.program_id(0) * BLOCK_COL
    cols = col_base + tl.arange(0, BLOCK_COL)[:, None]
    chunks = tl.arange(0, RBLOCK)[None, :]
    mask = (cols < 1536) & (chunks < 59)
    partials = tl.load(partial_ptr + chunks * 1536 + cols, mask=mask, other=0.0)
    sums = tl.sum(partials, axis=1)[:, None].to(tl.float32)
    rounded = _round_to_bf16_f32(sums)
    out_cols = tl.where(cols < 768, cols, cols + 768)
    tl.store(sum_ptr + out_cols, rounded, mask=cols < 1536)


# 96e55468: (T([128,12,197,64], bf16, stride=(151296,64,768,1)), T([128,12,197,64], bf16, stride=(151296,64,768,1)), T([128,12,197,64], bf16, stride=(151296,64,768,1)), ...)
@oracle_impl(
    hardware="B200",
    point="96e55468",
    K_BLOCK=1024,
    PARTIAL_COL_BLOCK=64,
    PARTIAL_RBLOCK=8,
    FINAL_COL_BLOCK=32,
    FINAL_RBLOCK=64,
    num_warps=4,
)
def oracle_forward(
    inputs,
    *,
    K_BLOCK: int,
    PARTIAL_COL_BLOCK: int,
    PARTIAL_RBLOCK: int,
    FINAL_COL_BLOCK: int,
    FINAL_RBLOCK: int,
    num_warps: int,
):
    q, k, v, _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3 = inputs
    del _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3

    qkv = torch.empty_strided((ROWS, COLS), (COLS, 1), device=q.device, dtype=torch.bfloat16)
    partials = torch.empty((PARTIAL_CHUNKS, Q_OR_V_COLS), device=q.device, dtype=torch.float32)
    sum_base = torch.empty_strided((COLS,), (1,), device=q.device, dtype=torch.float32)

    _materialize_qv_partial_sum_kernel[
        (PARTIAL_CHUNKS, triton.cdiv(Q_OR_V_COLS, PARTIAL_COL_BLOCK))
    ](
        q,
        v,
        qkv,
        partials,
        BLOCK_COL=PARTIAL_COL_BLOCK,
        RBLOCK=PARTIAL_RBLOCK,
        num_warps=num_warps,
    )
    _materialize_k_kernel[(triton.cdiv(ROWS * HEAD_COLS, K_BLOCK),)](
        k,
        qkv,
        total=ROWS * HEAD_COLS,
        BLOCK=K_BLOCK,
        num_warps=num_warps,
    )
    _final_qv_sum_kernel[(triton.cdiv(Q_OR_V_COLS, FINAL_COL_BLOCK),)](
        partials,
        sum_base,
        BLOCK_COL=FINAL_COL_BLOCK,
        RBLOCK=FINAL_RBLOCK,
        num_warps=num_warps,
    )

    qkv_t = torch.as_strided(qkv, (COLS, ROWS), (1, COLS), 0)
    sum_q = torch.as_strided(sum_base, (HEAD_COLS,), (1,), 0)
    sum_v = torch.as_strided(sum_base, (HEAD_COLS,), (1,), 2 * HEAD_COLS)
    return qkv, qkv_t, sum_q, sum_v
