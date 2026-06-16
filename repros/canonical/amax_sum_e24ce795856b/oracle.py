"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete Longformer bf16 sliding-window attention scope, using a Triton scatter-gather score kernel plus exact fp32 amax/sum, query-mask fill, dropout mask, and strided layout epilogue for all returned intermediates, whereas Inductor lowers the slice/scatter/permute assembly, generic reduction, RNG/dropout, padding, and layout views as separate regions; Inductor cannot do this today because scheduler/codegen has no Longformer scatter-gather attention pattern that fuses structured band assembly with the reduction epilogue and destination-layout scatter while preserving returned intermediates; the fix is NEW_PATTERN: add a Longformer sliding-window attention lowering that recognizes this banded score construction and emits a full-scope fused softmax/dropout/layout kernel."""

import struct

import torch
import torch._inductor.inductor_prims  # noqa: F401
import triton
import triton.language as tl

from oracle_harness import oracle_impl


BATCH = 8
SEQ = 1024
HEADS = 12
R = 513
CHUNK = 256
ROWS = BATCH * SEQ * HEADS
BLOCK = 1024
RNG_NUMEL = ROWS * R
CUDA_RAND_BLOCK = 256
CUDA_RAND_UNROLL = 4
CUDA_CURAND4_CALLS = 4
_RNG_STATE_CACHE = {"seed": 1, "base_offset": 0, "grid": 1}


@triton.jit
def _bf16_round_f32(x):
    return x.to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)


@triton.jit
def _load_skewed_bmm(bmm_ptr, bh, chunk, skew_row, skew_col, mask):
    linear = skew_row * 513 + skew_col
    src_row = linear // 512
    src_col = linear - src_row * 512
    valid = mask & (src_row < 512)
    safe_chunk = tl.where(valid, chunk, 0)
    safe_row = tl.where(valid, src_row, 0)
    safe_col = tl.where(valid, src_col, 0)
    offset = (bh * 3 + safe_chunk) * 262144 + safe_row * 512 + safe_col
    return tl.load(bmm_ptr + offset, mask=valid, other=0.0).to(tl.float32)


@triton.jit
def _longformer_scope_kernel(
    bmm_ptr,
    slice3_ptr,
    first_mask_ptr,
    edge_value_ptr,
    last_mask_ptr,
    bias_ptr,
    score_ptr,
    BLOCK_SIZE: tl.constexpr,
    R_SIZE: tl.constexpr,
    SEQ_SIZE: tl.constexpr,
    HEADS_SIZE: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    valid_cols = cols < R_SIZE

    head = row % HEADS_SIZE
    row_div_heads = row // HEADS_SIZE
    seq = row_div_heads % SEQ_SIZE
    batch = row_div_heads // SEQ_SIZE
    bh = batch * HEADS_SIZE + head
    chunk_id = seq // CHUNK_SIZE
    pos = seq - chunk_id * CHUNK_SIZE

    col_i32 = cols.to(tl.int32)
    from_right = col_i32 >= CHUNK_SIZE
    source_chunk_right = tl.minimum(chunk_id, 2)
    skew_row_right = tl.where(chunk_id == 3, pos + CHUNK_SIZE, pos)
    source_chunk_left = tl.where(chunk_id == 0, 0, chunk_id - 1)
    skew_row_left = tl.where(chunk_id == 0, pos - 1, pos + CHUNK_SIZE - 1)
    source_chunk = tl.where(from_right, source_chunk_right, source_chunk_left)
    skew_row = tl.where(from_right, skew_row_right, skew_row_left)
    skew_col = tl.where(from_right, col_i32 - CHUNK_SIZE, col_i32 + CHUNK_SIZE + 1)
    first_left_interior = (chunk_id == 0) & (pos > 0) & (col_i32 > 0) & (col_i32 < CHUNK_SIZE)
    has_bmm_source = from_right | (chunk_id != 0) | first_left_interior

    bmm_score = _load_skewed_bmm(
        bmm_ptr,
        bh,
        source_chunk,
        skew_row,
        skew_col,
        valid_cols & has_bmm_source,
    )

    safe_col = tl.minimum(col_i32, R_SIZE - 1)
    first_chunk_base = tl.load(
        slice3_ptr + bh * 525312 + pos * 513 + safe_col,
        mask=valid_cols,
        other=0.0,
    ).to(tl.float32)
    first_select = (chunk_id == 0) & ((pos == 0) | (col_i32 == 0)) & (col_i32 < CHUNK_SIZE)
    local_score = tl.where(first_select, first_chunk_base, bmm_score)

    first_col = tl.minimum(col_i32, 256)
    first_mask_offset = ((batch * CHUNK_SIZE + pos) * HEADS_SIZE + head) * 257 + first_col
    edge_value_offset = batch * 789504 + pos * 257 + head * 65792 + first_col
    first_mask = tl.load(
        first_mask_ptr + first_mask_offset,
        mask=(chunk_id == 0) & (col_i32 < 257),
        other=0,
    ) != 0
    first_value = tl.load(
        edge_value_ptr + edge_value_offset,
        mask=(chunk_id == 0) & (col_i32 < 257),
        other=0.0,
    ).to(tl.float32)
    local_score = tl.where((chunk_id == 0) & (col_i32 < 257) & first_mask, first_value, local_score)

    last_col = col_i32 - CHUNK_SIZE
    safe_last_col = tl.maximum(tl.minimum(last_col, 256), 0)
    last_mask_offset = ((batch * CHUNK_SIZE + pos) * HEADS_SIZE + head) * 257 + safe_last_col
    last_value_offset = batch * 789504 + pos * 257 + head * 65792 + safe_last_col
    last_mask = tl.load(
        last_mask_ptr + last_mask_offset,
        mask=(chunk_id == 3) & (col_i32 >= CHUNK_SIZE) & valid_cols,
        other=0,
    ) != 0
    last_value = tl.load(
        edge_value_ptr + last_value_offset,
        mask=(chunk_id == 3) & (col_i32 >= CHUNK_SIZE) & valid_cols,
        other=0.0,
    ).to(tl.float32)
    local_score = tl.where(
        (chunk_id == 3) & (col_i32 >= CHUNK_SIZE) & last_mask,
        last_value,
        local_score,
    )

    bias = tl.load(
        bias_ptr + (batch * SEQ_SIZE + seq) * R_SIZE + safe_col,
        mask=valid_cols,
        other=0.0,
    ).to(tl.float32)
    score = _bf16_round_f32(local_score + bias)

    score_offsets = (batch * HEADS_SIZE + head) * (SEQ_SIZE * R_SIZE) + seq * R_SIZE + safe_col
    tl.store(score_ptr + score_offsets, score, mask=valid_cols)


@triton.jit
def _cuda_rand_mask_kernel(
    mask_ptr,
    seed,
    base_offset,
    BLOCK_SIZE: tl.constexpr,
    N_SIZE: tl.constexpr,
    GRID_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    block_grid = GRID_SIZE * 256
    loop = offsets // (block_grid * 4)
    rem = offsets - loop * block_grid * 4
    lane = rem // block_grid
    thread = rem - lane * block_grid

    c0 = ((base_offset // 4) + loop).to(tl.uint32)
    zero = tl.full((BLOCK_SIZE,), 0, dtype=tl.uint32)
    i0, i1, i2, i3 = tl.philox(seed, c0, zero, thread.to(tl.uint32), zero)
    u0 = i0.to(tl.uint32).to(tl.float32) * 2.3283064e-10
    u1 = i1.to(tl.uint32).to(tl.float32) * 2.3283064e-10
    u2 = i2.to(tl.uint32).to(tl.float32) * 2.3283064e-10
    u3 = i3.to(tl.uint32).to(tl.float32) * 2.3283064e-10
    value = tl.where(lane == 0, u0, tl.where(lane == 1, u1, tl.where(lane == 2, u2, u3)))
    value = tl.where(value == 1.0, 0.0, value)
    value_bf16 = value.to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
    threshold = tl.full((BLOCK_SIZE,), 0.1, tl.float32).to(
        tl.bfloat16,
        fp_downcast_rounding="rtne",
    ).to(tl.float32)
    tl.store(mask_ptr + offsets, value_bf16 > threshold, mask=offsets < N_SIZE)


def _cuda_rand_params(device):
    if not torch.cuda.is_current_stream_capturing():
        state = torch.cuda.get_rng_state(device)
        state_bytes = bytearray(state.cpu().numpy().tobytes())
        seed, offset = struct.unpack_from("<QQ", state_bytes, 0)
        props = torch.cuda.get_device_properties(device)
        grid = min(
            (RNG_NUMEL + CUDA_RAND_BLOCK - 1) // CUDA_RAND_BLOCK,
            props.multi_processor_count
            * (props.max_threads_per_multi_processor // CUDA_RAND_BLOCK),
        )
        jump = (
            (RNG_NUMEL - 1) // (CUDA_RAND_BLOCK * grid * CUDA_RAND_UNROLL) + 1
        ) * CUDA_CURAND4_CALLS
        _RNG_STATE_CACHE["seed"] = int(seed)
        _RNG_STATE_CACHE["base_offset"] = int(offset - jump) if offset >= jump else 0
        _RNG_STATE_CACHE["grid"] = int(grid)
    return (
        _RNG_STATE_CACHE["seed"],
        _RNG_STATE_CACHE["base_offset"],
        _RNG_STATE_CACHE["grid"],
    )


@oracle_impl(hardware="B200", point="b64f0e8a", num_warps=4)
def oracle_forward(inputs, *, num_warps: int):
    (
        arg0_1,
        _arg1_1,
        arg2_1,
        _arg3_1,
        arg4_1,
        arg5_1,
        arg6_1,
        arg7_1,
        arg8_1,
        arg9_1,
        _arg10_1,
        *_shape_params,
    ) = inputs

    device = arg0_1.device
    scores_base = torch.empty_strided(
        (BATCH, HEADS, SEQ, R),
        (HEADS * SEQ * R, SEQ * R, R, 1),
        device=device,
        dtype=torch.bfloat16,
    )
    _longformer_scope_kernel[(ROWS,)](
        arg0_1,
        arg2_1,
        arg4_1,
        arg5_1,
        arg6_1,
        arg7_1,
        scores_base,
        BLOCK_SIZE=BLOCK,
        R_SIZE=R,
        SEQ_SIZE=SEQ,
        HEADS_SIZE=HEADS,
        CHUNK_SIZE=CHUNK,
        num_warps=num_warps,
    )
    scores = scores_base.permute(0, 2, 1, 3)
    score_f32 = torch.ops.prims.convert_element_type.default(scores, torch.float32)
    max_out = torch.ops.aten.amax.default(score_f32, [-1], True)
    shifted = torch.ops.aten.sub.Tensor(score_f32, max_out)
    exp = torch.ops.aten.exp.default(shifted)
    sum_out = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
    div = torch.ops.aten.div.Tensor(exp, sum_out)
    masked = torch.ops.aten.where.self(arg8_1, arg9_1, div)
    masked_bf16 = torch.ops.prims.convert_element_type.default(masked, torch.bfloat16)

    gt_out = torch.empty((BATCH, SEQ, HEADS, R), device=device, dtype=torch.bool)
    rng_seed, rng_base_offset, rng_grid = _cuda_rand_params(device)
    _cuda_rand_mask_kernel[(triton.cdiv(RNG_NUMEL, BLOCK),)](
        gt_out,
        rng_seed,
        rng_base_offset,
        BLOCK_SIZE=BLOCK,
        N_SIZE=RNG_NUMEL,
        GRID_SIZE=rng_grid,
        num_warps=num_warps,
    )
    dropped = torch.ops.aten.mul.Tensor(gt_out, masked_bf16)
    dropped = torch.ops.aten.mul.Tensor(dropped, 1.1111111111111112)
    permuted = torch.ops.aten.permute.default(dropped, [0, 2, 1, 3])
    cloned = torch.ops.aten.clone.default(permuted, memory_format=torch.contiguous_format)
    view_6 = torch.ops.aten.view.default(cloned, [96, 4, 256, 513])
    padded = torch.ops.aten.constant_pad_nd.default(view_6, [0, 257], 0.0)
    view_7 = torch.ops.aten.view.default(padded, [96, 4, -1])
    slice_18 = torch.ops.aten.slice.Tensor(view_7, 2, 0, -256)
    view_8 = torch.ops.aten.view.default(slice_18, [96, 4, 256, 769])
    slice_19 = torch.ops.aten.slice.Tensor(view_8, 3, 0, -1)
    unsqueezed = torch.ops.aten.unsqueeze.default(slice_19, 4)
    layout = torch.ops.aten.view.default(unsqueezed, [384, 256, 768])
    return scores, max_out, sum_out, gt_out, layout, layout.permute(0, 2, 1)
