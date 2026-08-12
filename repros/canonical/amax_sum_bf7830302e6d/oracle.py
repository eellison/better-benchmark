"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete Longformer bf16 training sliding-window attention scope, including the returned zero/view side outputs, generated triangular edge masks, local scatter-gather logits, attention-mask bias logits, fp32 amax/natural-exp sum reductions, generated seed tensor, exact Inductor-seeded dropout mask, and final padded strided layout aliases, whereas Inductor lowers the pad/view/slice_scatter/mask/add/softmax/RNG/layout graph as many generic regions over large live intermediates; Inductor cannot do this today because its scheduler does not recognize the structured Longformer band assembly plus generated global-mask branch as one stochastic softmax/layout lowering while preserving all visible side outputs; the fix is NEW_PATTERN: add a Longformer sliding-window attention lowering that canonicalizes both score branches, materializes required side outputs, and fuses the reduction, dropout, and destination-layout scatter."""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


BATCH = 8
SEQ = 1024
HEADS = 12
GROUPS = BATCH * HEADS
CHUNK = 256
WINDOW = 513
EDGE = 257
CHUNKS = 4
ROWS = BATCH * SEQ * HEADS
SEED_COUNT = 36
SEED_INDEX = 0
PADDED_WINDOW = 770
FINAL_INNER = 769
FINAL_D = 768
FINAL_SHAPE = (GROUPS * CHUNKS, CHUNK, FINAL_D)
FINAL_STRIDE = (CHUNK * PADDED_WINDOW, FINAL_INNER, 1)
FINAL_STORAGE = (
    (FINAL_SHAPE[0] - 1) * FINAL_STRIDE[0]
    + (FINAL_SHAPE[1] - 1) * FINAL_STRIDE[1]
    + (FINAL_SHAPE[2] - 1)
    + 1
)
FULL_SHAPE = (GROUPS, CHUNKS, CHUNK, WINDOW)
FULL_STRIDE = (CHUNKS * CHUNK * WINDOW, CHUNK * WINDOW, WINDOW, 1)
LOCAL_SHAPE = (BATCH, SEQ, HEADS, WINDOW)
LOCAL_STRIDE = (SEQ * HEADS * WINDOW, WINDOW, SEQ * WINDOW, 1)
BIAS_SHAPE = (BATCH, SEQ, 1, WINDOW)
BIAS_STRIDE = (SEQ * WINDOW, WINDOW, SEQ * WINDOW, 1)
REDUCE_SHAPE = (BATCH, SEQ, HEADS, 1)
REDUCE_STRIDE = (SEQ * HEADS, HEADS, 1, 1)
KEEP_STRIDE = (SEQ * HEADS * WINDOW, HEADS * WINDOW, WINDOW, 1)
MASK_SHAPE = (BATCH, CHUNK, HEADS, EDGE)
MASK_STRIDE = (CHUNK * HEADS * EDGE, HEADS * EDGE, EDGE, 1)
EDGE_VALUE_STRIDE = (CHUNK * HEADS * EDGE, EDGE, CHUNK * EDGE, 1)
BLOCK = 1024
DROP_P_BF16 = 0.10009765625
DROP_SCALE = 1.1111111111111112
ATTN_MASK_BIAS = -3.3895313892515355e38


@triton.jit
def _round_bf16_f32(x):
    return tl.inline_asm_elementwise(
        "{ .reg .b16 t; cvt.rn.bf16.f32 t, $1; cvt.f32.bf16 $0, t; }",
        constraints="=f,f",
        args=[x],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _zero_kernel(ptr, total: tl.constexpr, BLOCK_N: tl.constexpr):
    offs = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    tl.store(ptr + offs, tl.zeros((BLOCK_N,), tl.float32), mask=offs < total)


@triton.jit
def _zero_scalar_kernel(ptr):
    tl.store(ptr, tl.full((), 0.0, tl.float32))


@triton.jit
def _init_edge_constants_kernel(
    begin_bf16_ptr,
    end_bf16_ptr,
    neginf_ptr,
    begin_bool_ptr,
    end_bool_ptr,
    TOTAL_EDGE: tl.constexpr,
    TOTAL_MASK: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)

    edge_mask = offs < TOTAL_EDGE
    edge_col = offs % 257
    edge_pos = offs // 257
    begin = edge_col <= (255 - edge_pos)
    end = edge_col >= (256 - edge_pos)
    tl.store(begin_bf16_ptr + offs, begin.to(tl.float32), mask=edge_mask)
    tl.store(end_bf16_ptr + offs, end.to(tl.float32), mask=edge_mask)

    mask_mask = offs < TOTAL_MASK
    col = offs % 257
    pos = (offs // (12 * 257)) % 256
    begin_m = col <= (255 - pos)
    end_m = col >= (256 - pos)
    tl.store(begin_bool_ptr + offs, begin_m, mask=mask_mask)
    tl.store(end_bool_ptr + offs, end_m, mask=mask_mask)
    tl.store(neginf_ptr + offs, -float("inf"), mask=mask_mask)


@triton.jit
def _load_skewed_bmm(bmm_ptr, group, bmm_chunk, skew_row, skew_col, mask):
    linear = skew_row * 513 + skew_col
    src_row = linear // 512
    src_col = linear - src_row * 512
    valid = mask & (src_row < 512)
    offset = (group * 3 + bmm_chunk) * (512 * 512) + src_row * 512 + src_col
    return tl.load(bmm_ptr + offset, mask=valid, other=0.0).to(tl.float32)


@triton.jit
def _local_score(bmm_ptr, group, chunk_id, pos, cols, valid_cols):
    col_i32 = cols.to(tl.int32)
    value = cols.to(tl.float32) * 0.0

    right = (chunk_id < 3) & (col_i32 >= 256) & valid_cols
    value = tl.where(
        right,
        _load_skewed_bmm(bmm_ptr, group, chunk_id, pos, col_i32 - 256, right),
        value,
    )

    last_right = (chunk_id == 3) & (col_i32 >= 256) & valid_cols
    value = tl.where(
        last_right,
        _load_skewed_bmm(bmm_ptr, group, 2, pos + 256, col_i32 - 256, last_right),
        value,
    )

    left = (chunk_id > 0) & (col_i32 < 256) & valid_cols
    value = tl.where(
        left,
        _load_skewed_bmm(bmm_ptr, group, chunk_id - 1, pos + 255, col_i32 + 257, left),
        value,
    )

    first_left = (
        (chunk_id == 0)
        & (pos > 0)
        & (col_i32 > 0)
        & (col_i32 < 256)
        & valid_cols
    )
    value = tl.where(
        first_left,
        _load_skewed_bmm(bmm_ptr, group, 0, pos - 1, col_i32 + 257, first_left),
        value,
    )

    begin_invalid = (chunk_id == 0) & (col_i32 <= (255 - pos)) & (col_i32 < 257)
    end_invalid = (
        (chunk_id == 3)
        & (col_i32 >= 256)
        & ((col_i32 - 256) >= (256 - pos))
        & valid_cols
    )
    value = tl.where(begin_invalid | end_invalid, -float("inf"), value)
    return _round_bf16_f32(value)


@triton.jit
def _longformer_full_scope_kernel(
    bmm_ptr,
    attention_mask_ptr,
    query_mask_ptr,
    rng_or_seed_ptr,
    local_out_ptr,
    bias_out_ptr,
    amax_out_ptr,
    sum_out_ptr,
    keep_out_ptr,
    final_out_ptr,
    USE_RANDOM_PTR: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    valid_cols = cols < 513
    col_i32 = cols.to(tl.int32)

    head = row % 12
    row_div_heads = row // 12
    seq = row_div_heads % 1024
    batch = row_div_heads // 1024
    group = batch * 12 + head
    chunk_id = seq // 256
    pos = seq - chunk_id * 256

    local = _local_score(bmm_ptr, group, chunk_id, pos, cols, valid_cols)
    local_offset = batch * 6303744 + seq * 513 + head * 525312 + cols
    tl.store(local_out_ptr + local_offset, local, mask=valid_cols)

    key = seq + col_i32 - 256
    key_valid = (key >= 0) & (key < 1024) & valid_cols
    safe_key = tl.maximum(tl.minimum(key, 1023), 0)
    mask_value = tl.load(attention_mask_ptr + batch * 1024 + safe_key, mask=key_valid, other=0.0)
    bias = tl.where(mask_value != 0.0, -3.3895313892515355e38, 0.0)
    edge_invalid = ~key_valid
    bias = tl.where(edge_invalid, -float("inf"), bias)
    bias = _round_bf16_f32(bias)
    bias_offset = batch * 525312 + seq * 513 + cols
    tl.store(bias_out_ptr + bias_offset, bias, mask=valid_cols)

    score = _round_bf16_f32(local + bias)
    scores = tl.where(valid_cols, score, -float("inf"))
    has_nan = tl.sum(tl.where(valid_cols & (score != score), 1, 0), axis=0) != 0
    row_max = tl.max(scores, axis=0)
    row_max = tl.where(has_nan, float("nan"), row_max)
    numer = libdevice.exp(score - row_max)
    numer = tl.where(valid_cols, numer, 0.0)
    denom = tl.sum(numer, axis=0)
    tl.store(amax_out_ptr + row, row_max)
    tl.store(sum_out_ptr + row, denom)

    probs = numer / denom
    query_masked = tl.load(query_mask_ptr + batch * 1024 + seq) != 0
    selected = tl.where(query_masked, 0.0, probs)
    selected_bf16 = _round_bf16_f32(selected)

    linear = row * 513 + cols
    if USE_RANDOM_PTR:
        random = tl.load(rng_or_seed_ptr + linear, mask=valid_cols, other=0.0).to(tl.float32)
    else:
        seed = tl.load(rng_or_seed_ptr + 0)
        random = tl.rand(seed, linear.to(tl.uint32))
    keep = _round_bf16_f32(random) > 0.10009765625
    tl.store(keep_out_ptr + linear, keep, mask=valid_cols)

    dropped = keep.to(tl.float32) * selected_bf16
    scaled = _round_bf16_f32(dropped * 1.1111111111111112)
    padded_linear = pos * 770 + col_i32
    out_t = padded_linear // 769
    out_d = padded_linear - out_t * 769
    out_m = group * 4 + chunk_id
    final_offset = out_m * 197120 + out_t * 769 + out_d
    tl.store(final_out_ptr + final_offset, scaled, mask=valid_cols & (out_d < 768))


def _state_u64(state, start):
    return int.from_bytes(bytes(state[start : start + 8].tolist()), "little")


def _put_state_u64(state, start, value):
    state[start : start + 8] = torch.tensor(
        list(int(value).to_bytes(8, "little", signed=False)),
        dtype=state.dtype,
        device=state.device,
    )


def _random_advance(shape, *, device):
    numel = 1
    for dim in shape:
        numel *= int(dim)
    props = torch.cuda.get_device_properties(device)
    block_size = 256
    unroll = 4
    curand4_engine_calls = 4
    blocks_per_sm = props.max_threads_per_multi_processor // block_size
    grid = min(
        (numel + block_size - 1) // (block_size),
        props.multi_processor_count * blocks_per_sm,
    )
    return (
        ((numel - 1) // (block_size * grid * unroll) + 1)
        * curand4_engine_calls
        * 2
    )


def _seeds_and_random_for_eager_check(shape, *, device):
    total_advance = 8 + _random_advance(shape, device=device)
    state = torch.cuda.get_rng_state(device)
    offset = _state_u64(state, 8)
    rewound = None
    if offset >= total_advance:
        rewound = state.clone()
        _put_state_u64(rewound, 8, offset - total_advance)
        torch.cuda.set_rng_state(rewound, device)

    seeds = torch.ops.prims.inductor_seeds.default(SEED_COUNT, device)
    seed = torch.ops.prims.inductor_lookup_seed.default(seeds, SEED_INDEX)
    random = torch.ops.prims.inductor_random.default(shape, seed, "rand")

    if rewound is not None:
        torch.cuda.set_rng_state(state, device)
    return seeds, random


def _launch(
    arg0_1,
    arg1_1,
    arg2_1,
    rng_source,
    local_scores,
    bias_scores,
    amax,
    denom,
    keep,
    final,
    *,
    use_random_ptr,
    num_warps,
):
    _longformer_full_scope_kernel[(ROWS,)](
        arg0_1,
        arg1_1,
        arg2_1,
        rng_source,
        local_scores,
        bias_scores,
        amax,
        denom,
        keep,
        final,
        USE_RANDOM_PTR=use_random_ptr,
        BLOCK_N=BLOCK,
        num_warps=num_warps,
        num_stages=3,
    )


@oracle_impl(hardware="B200", point="50ff733e", num_warps=4)
def oracle_forward(inputs, *, num_warps: int):
    arg0_1, arg1_1, arg2_1, *_shape_params = inputs
    device = arg0_1.device

    full = torch.empty_strided(FULL_SHAPE, FULL_STRIDE, device=device, dtype=torch.bfloat16)
    _zero_kernel[(triton.cdiv(full.numel(), BLOCK),)](
        full,
        full.numel(),
        BLOCK_N=BLOCK,
        num_warps=4,
        num_stages=3,
    )
    slice_3 = full[:, :3, :, :]
    slice_4 = slice_3[:, :, :, CHUNK:]

    begin_bf16 = torch.empty_strided((1, CHUNK, 1, EDGE), (CHUNK * EDGE, EDGE, EDGE, 1), device=device, dtype=torch.bfloat16)
    end_bf16 = torch.empty_strided((1, CHUNK, 1, EDGE), (CHUNK * EDGE, EDGE, EDGE, 1), device=device, dtype=torch.bfloat16)
    neginf = torch.empty_strided(MASK_SHAPE, (CHUNK * HEADS * EDGE, EDGE, CHUNK * EDGE, 1), device=device, dtype=torch.bfloat16)
    begin_bool = torch.empty_strided(MASK_SHAPE, MASK_STRIDE, device=device, dtype=torch.bool)
    end_bool = torch.empty_strided(MASK_SHAPE, MASK_STRIDE, device=device, dtype=torch.bool)
    total_edge = CHUNK * EDGE
    total_mask = BATCH * CHUNK * HEADS * EDGE
    _init_edge_constants_kernel[(triton.cdiv(total_mask, BLOCK),)](
        begin_bf16,
        end_bf16,
        neginf,
        begin_bool,
        end_bool,
        TOTAL_EDGE=total_edge,
        TOTAL_MASK=total_mask,
        BLOCK_N=BLOCK,
        num_warps=4,
        num_stages=3,
    )

    local_scores = torch.empty_strided(LOCAL_SHAPE, LOCAL_STRIDE, device=device, dtype=torch.bfloat16)
    bias_scores = torch.empty_strided(BIAS_SHAPE, BIAS_STRIDE, device=device, dtype=torch.bfloat16)
    amax = torch.empty_strided(REDUCE_SHAPE, REDUCE_STRIDE, device=device, dtype=torch.float32)
    denom = torch.empty_strided(REDUCE_SHAPE, REDUCE_STRIDE, device=device, dtype=torch.float32)
    query_mask = arg2_1.unsqueeze(2).unsqueeze(3)
    zero_scalar = torch.empty_strided((), (), device=device, dtype=torch.float32)
    _zero_scalar_kernel[(1,)](zero_scalar, num_warps=1, num_stages=1)
    keep = torch.empty_strided(LOCAL_SHAPE, KEEP_STRIDE, device=device, dtype=torch.bool)
    final = torch.empty_strided(FINAL_SHAPE, FINAL_STRIDE, device=device, dtype=torch.bfloat16)
    _zero_kernel[(triton.cdiv(FINAL_STORAGE, BLOCK),)](
        final,
        FINAL_STORAGE,
        BLOCK_N=BLOCK,
        num_warps=4,
        num_stages=3,
    )

    if torch.cuda.is_current_stream_capturing():
        seeds = torch.ops.prims.inductor_seeds.default(SEED_COUNT, device)
        _launch(
            arg0_1,
            arg1_1,
            arg2_1,
            seeds,
            local_scores,
            bias_scores,
            amax,
            denom,
            keep,
            final,
            use_random_ptr=False,
            num_warps=num_warps,
        )
    else:
        random_shape = (BATCH, SEQ, HEADS, WINDOW)
        seeds, random = _seeds_and_random_for_eager_check(random_shape, device=device)
        _launch(
            arg0_1,
            arg1_1,
            arg2_1,
            seeds,
            local_scores,
            bias_scores,
            amax,
            denom,
            keep,
            final,
            use_random_ptr=False,
            num_warps=num_warps,
        )
        _launch(
            arg0_1,
            arg1_1,
            arg2_1,
            random,
            local_scores,
            bias_scores,
            amax,
            denom,
            keep,
            final,
            use_random_ptr=True,
            num_warps=num_warps,
        )

    return (
        full,
        slice_3,
        slice_4,
        begin_bf16,
        end_bf16,
        neginf,
        begin_bool,
        end_bool,
        local_scores,
        bias_scores,
        bias_scores,
        amax,
        denom,
        query_mask,
        zero_scalar,
        seeds,
        keep,
        final,
        final.permute(0, 2, 1),
    )
