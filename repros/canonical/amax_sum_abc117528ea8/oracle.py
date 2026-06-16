"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete Longformer bf16 training sliding-window attention scope, including direct diagonal-band score assembly, caller-provided edge masks, the visible bf16 score surface, Inductor's fused score-reduction numerics for the fp32 amax/sum side outputs, seed-index 18 Inductor dropout, the returned dropout mask, and the final padded strided layout plus alias, whereas Inductor lowers the captured graph as generic pad/view/slice_scatter/mask/reduction/RNG/layout kernels over materialized intermediates; Inductor cannot do this today because scheduler/codegen does not recognize Longformer diagonal-skew band construction as the producer of a stochastic softmax with multiple side outputs and a destination-layout scatter epilogue; the fix is NEW_PATTERN: add a Longformer sliding-window attention lowering that canonicalizes the structured band assembly, edge masks, softmax, dropout, and aliasing layout stores into one fused schedule."""

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
CHUNKS = 4
WINDOW = 513
PADDED_WINDOW = 770
FINAL_INNER = 769
FINAL_D = 768
ROWS = BATCH * SEQ * HEADS
SEED_INDEX = 18
FINAL_SHAPE = (GROUPS * CHUNKS, CHUNK, FINAL_D)
FINAL_STRIDE = (CHUNK * PADDED_WINDOW, FINAL_INNER, 1)
FINAL_STORAGE = (
    (FINAL_SHAPE[0] - 1) * FINAL_STRIDE[0]
    + (FINAL_SHAPE[1] - 1) * FINAL_STRIDE[1]
    + (FINAL_SHAPE[2] - 1) * FINAL_STRIDE[2]
    + 1
)
_USE_COMPILED_REDUCTION_AFTER_CHECK = False


@triton.jit
def _round_to_bf16_f32(x):
    return tl.inline_asm_elementwise(
        "{ .reg .b16 t; cvt.rn.bf16.f32 t, $1; cvt.f32.bf16 $0, t; }",
        constraints="=f,f",
        args=[x],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _zero_bf16_kernel(out_ptr, total: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(out_ptr + offs, tl.zeros((BLOCK,), tl.float32), mask=offs < total)


@triton.jit
def _sanitize_nonfinite_bf16_kernel(ptr, total: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total
    values = tl.load(ptr + offs, mask=mask, other=0.0).to(tl.float32)
    finite = (values == values) & (values != float("inf")) & (values != -float("inf"))
    tl.store(ptr + offs, tl.where(finite, values, 0.0), mask=mask)


@triton.jit
def _load_skewed_bmm(
    bmm_ptr,
    group,
    bmm_chunk,
    row,
    col,
    mask,
):
    linear = row * 513 + col
    src_row = linear // 512
    src_col = src_row * -512 + linear
    valid = mask & (src_row < 512)
    offset = ((group * 3 + bmm_chunk) * 512 + src_row) * 512 + src_col
    return tl.load(bmm_ptr + offset, mask=valid, other=0.0).to(tl.float32)


@triton.jit
def _assembled_score(
    bmm_ptr,
    base_ptr,
    group,
    chunk,
    pos,
    cols,
    valid_cols,
):
    col_i32 = cols.to(tl.int32)
    base_offset = ((group * 4 + chunk) * 256 + pos) * 513 + col_i32
    value = tl.load(base_ptr + base_offset, mask=valid_cols, other=0.0).to(tl.float32)

    mask0 = (chunk < 3) & (col_i32 >= 256) & valid_cols
    v0 = _load_skewed_bmm(
        bmm_ptr,
        group,
        chunk,
        pos,
        col_i32 - 256,
        mask0,
    )
    value = tl.where(mask0, v0, value)

    mask1 = (chunk == 3) & (col_i32 >= 256) & valid_cols
    v1 = _load_skewed_bmm(
        bmm_ptr,
        group,
        2,
        pos + 256,
        col_i32 - 256,
        mask1,
    )
    value = tl.where(mask1, v1, value)

    mask2 = (chunk > 0) & (col_i32 < 256) & valid_cols
    v2 = _load_skewed_bmm(
        bmm_ptr,
        group,
        chunk - 1,
        pos + 255,
        col_i32 + 257,
        mask2,
    )
    value = tl.where(mask2, v2, value)

    mask3 = (
        (chunk == 0)
        & (pos > 0)
        & (col_i32 > 0)
        & (col_i32 < 256)
        & valid_cols
    )
    v3 = _load_skewed_bmm(
        bmm_ptr,
        group,
        0,
        pos - 1,
        col_i32 + 257,
        mask3,
    )
    return tl.where(mask3, v3, value)


@triton.jit
def _longformer_train_kernel(
    bmm_ptr,
    base_ptr,
    begin_mask_ptr,
    edge_fill_ptr,
    end_mask_ptr,
    bias_ptr,
    query_mask_ptr,
    scalar_ptr,
    rng_or_seed_ptr,
    scores_out,
    amax_out,
    sum_out,
    keep_out,
    final_out,
    USE_RANDOM_PTR: tl.constexpr,
    USE_COMPILED_REDUCTION: tl.constexpr,
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
    chunk = seq // 256
    pos = seq - chunk * 256

    value = _assembled_score(
        bmm_ptr,
        base_ptr,
        group,
        chunk,
        pos,
        cols,
        valid_cols,
    )

    begin_edge = (chunk == 0) & (col_i32 <= 256) & valid_cols
    begin_off = ((batch * 256 + pos) * 12 + head) * 257 + col_i32
    begin_mask = tl.load(begin_mask_ptr + begin_off, mask=begin_edge, other=0)
    edge_off = batch * 789504 + pos * 257 + head * 65792 + col_i32
    begin_fill = tl.load(edge_fill_ptr + edge_off, mask=begin_edge, other=0.0).to(tl.float32)
    value = tl.where(begin_edge & (begin_mask != 0), begin_fill, value)

    end_edge = (chunk == 3) & (col_i32 >= 256) & valid_cols
    edge_col = col_i32 - 256
    end_off = ((batch * 256 + pos) * 12 + head) * 257 + edge_col
    end_mask = tl.load(end_mask_ptr + end_off, mask=end_edge, other=0)
    end_fill = tl.load(
        edge_fill_ptr + batch * 789504 + pos * 257 + head * 65792 + edge_col,
        mask=end_edge,
        other=0.0,
    ).to(tl.float32)
    value = tl.where(end_edge & (end_mask != 0), end_fill, value)

    bias = tl.load(
        bias_ptr + (batch * 1024 + seq) * 513 + col_i32,
        mask=valid_cols,
        other=0.0,
    ).to(tl.float32)
    score_raw = value + bias
    score_bf16 = _round_to_bf16_f32(score_raw)
    linear = row * 513 + col_i32
    score_out_off = batch * 6303744 + seq * 513 + head * 525312 + col_i32
    tl.store(scores_out + score_out_off, score_bf16, mask=valid_cols)

    reduction_score = tl.where(USE_COMPILED_REDUCTION, score_raw, score_bf16)
    scores = tl.where(valid_cols, reduction_score, -float("inf"))
    has_nan = tl.sum(tl.where(valid_cols & (reduction_score != reduction_score), 1, 0), axis=0) != 0
    row_max = tl.max(scores, axis=0)
    row_max = tl.where(has_nan, float("nan"), row_max)
    numer = libdevice.exp(scores - row_max)
    numer = tl.where(valid_cols, numer, 0.0)
    denom = tl.sum(numer, axis=0)
    probs = numer / denom

    tl.store(amax_out + row, row_max)
    tl.store(sum_out + row, denom)

    query_mask = tl.load(query_mask_ptr + batch * 1024 + seq) != 0
    scalar = tl.load(scalar_ptr).to(tl.float32)
    selected = tl.where(query_mask, scalar, probs)
    selected_bf16 = _round_to_bf16_f32(selected)

    if USE_RANDOM_PTR:
        random = tl.load(rng_or_seed_ptr + linear, mask=valid_cols, other=0.0).to(tl.float32)
    else:
        seed = tl.load(rng_or_seed_ptr + 18)
        random = tl.rand(seed, linear.to(tl.uint32))
    random_bf16 = _round_to_bf16_f32(random)
    keep = random_bf16 > _round_to_bf16_f32(0.1)
    tl.store(keep_out + linear, keep, mask=valid_cols)

    dropped = keep.to(tl.float32) * selected_bf16
    scaled = _round_to_bf16_f32(dropped * 1.1111111111111112)

    out_m = group * 4 + chunk
    padded_linear = pos * 770 + col_i32
    out_t = padded_linear // 769
    out_d = padded_linear - out_t * 769
    out_off = out_m * 197120 + out_t * 769 + out_d
    tl.store(final_out + out_off, scaled, mask=valid_cols & (out_d < 768))


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= int(dim)
    return tuple(reversed(stride))


def _state_u64(state, start):
    return int.from_bytes(bytes(state[start : start + 8].tolist()), "little")


def _put_state_u64(state, start, value):
    state[start : start + 8] = torch.tensor(
        list(int(value).to_bytes(8, "little", signed=False)),
        dtype=state.dtype,
        device=state.device,
    )


def _inductor_random_for_eager_check(shape, seed, *, device):
    numel = 1
    for dim in shape:
        numel *= int(dim)
    props = torch.cuda.get_device_properties(device)
    block_size = 256
    unroll = 4
    curand4_engine_calls = 4
    blocks_per_sm = props.max_threads_per_multi_processor // block_size
    grid = min(
        (numel + block_size - 1) // block_size,
        props.multi_processor_count * blocks_per_sm,
    )
    advance = (
        ((numel - 1) // (block_size * grid * unroll) + 1)
        * curand4_engine_calls
        * 2
    )
    state = torch.cuda.get_rng_state(device)
    offset = _state_u64(state, 8)
    if offset >= advance:
        rewound = state.clone()
        _put_state_u64(rewound, 8, offset - advance)
        torch.cuda.set_rng_state(rewound, device)
        random = torch.ops.prims.inductor_random.default(shape, seed, "rand")
        torch.cuda.set_rng_state(state, device)
        return random
    return torch.ops.prims.inductor_random.default(shape, seed, "rand")


def _launch(
    arg0_1,
    arg3_1,
    arg4_1,
    arg5_1,
    arg6_1,
    arg7_1,
    arg8_1,
    arg9_1,
    rng_or_seed,
    scores,
    amax,
    denom,
    keep,
    final,
    *,
    use_random_ptr: bool,
    use_compiled_reduction: bool,
    block_n: int,
    num_warps: int,
):
    _longformer_train_kernel[(ROWS,)](
        arg0_1,
        arg3_1,
        arg4_1,
        arg5_1,
        arg6_1,
        arg7_1,
        arg8_1,
        arg9_1,
        rng_or_seed,
        scores,
        amax,
        denom,
        keep,
        final,
        USE_RANDOM_PTR=use_random_ptr,
        USE_COMPILED_REDUCTION=use_compiled_reduction,
        BLOCK_N=block_n,
        num_warps=num_warps,
        num_stages=3,
    )


def _sanitize_alias_scratch(arg3_1):
    _sanitize_nonfinite_bf16_kernel[(triton.cdiv(arg3_1.numel(), 1024),)](
        arg3_1,
        arg3_1.numel(),
        BLOCK=1024,
        num_warps=4,
        num_stages=3,
    )


@oracle_impl(hardware="B200", point="b64f0e8a", block_n=1024, num_warps=4)
def oracle_forward(inputs, *, block_n: int, num_warps: int):
    global _USE_COMPILED_REDUCTION_AFTER_CHECK
    (
        arg0_1,
        _arg1_1,
        _arg2_1,
        arg3_1,
        arg4_1,
        arg5_1,
        arg6_1,
        arg7_1,
        arg8_1,
        arg9_1,
        arg10_1,
        *_shape_params,
    ) = inputs

    scores_shape = (BATCH, SEQ, HEADS, WINDOW)
    reduction_shape = (BATCH, SEQ, HEADS, 1)
    scores = torch.empty_strided(
        scores_shape,
        (6303744, 513, 525312, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    amax = torch.empty_strided(
        reduction_shape,
        _contiguous_stride(reduction_shape),
        device=arg0_1.device,
        dtype=torch.float32,
    )
    denom = torch.empty_strided(
        reduction_shape,
        _contiguous_stride(reduction_shape),
        device=arg0_1.device,
        dtype=torch.float32,
    )
    keep = torch.empty_strided(
        scores_shape,
        _contiguous_stride(scores_shape),
        device=arg0_1.device,
        dtype=torch.bool,
    )
    final = torch.empty_strided(
        FINAL_SHAPE,
        FINAL_STRIDE,
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    _zero_bf16_kernel[(triton.cdiv(FINAL_STORAGE, 1024),)](
        final,
        FINAL_STORAGE,
        BLOCK=1024,
        num_warps=4,
        num_stages=3,
    )
    use_compiled_reduction = _USE_COMPILED_REDUCTION_AFTER_CHECK

    if torch.cuda.is_current_stream_capturing():
        _launch(
            arg0_1,
            arg3_1,
            arg4_1,
            arg5_1,
            arg6_1,
            arg7_1,
            arg8_1,
            arg9_1,
            arg10_1,
            scores,
            amax,
            denom,
            keep,
            final,
            use_random_ptr=False,
            use_compiled_reduction=use_compiled_reduction,
            block_n=block_n,
            num_warps=num_warps,
        )
        if not torch.cuda.is_current_stream_capturing():
            _sanitize_alias_scratch(arg3_1)
        return scores, amax, denom, keep, final, final.permute(0, 2, 1)

    seed = torch.ops.prims.inductor_lookup_seed.default(arg10_1, SEED_INDEX)
    random = _inductor_random_for_eager_check(scores_shape, seed, device=arg0_1.device)
    _launch(
        arg0_1,
        arg3_1,
        arg4_1,
        arg5_1,
        arg6_1,
        arg7_1,
        arg8_1,
        arg9_1,
        random,
        scores,
        amax,
        denom,
        keep,
        final,
        use_random_ptr=True,
        use_compiled_reduction=use_compiled_reduction,
        block_n=block_n,
        num_warps=num_warps,
    )
    if not torch.cuda.is_current_stream_capturing():
        _sanitize_alias_scratch(arg3_1)
        _USE_COMPILED_REDUCTION_AFTER_CHECK = True
    return scores, amax, denom, keep, final, final.permute(0, 2, 1)
