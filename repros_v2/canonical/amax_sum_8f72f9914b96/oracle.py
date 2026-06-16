"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the full Longformer sliding-window attention training scope, including skewed chunk score assembly, first/last chunk scatter overrides, global bias, the returned bf16 logits view, fp32 amax and exp-sum reductions, query-mask fill, exact Inductor-seeded dropout mask, and both final padded layout views, whereas Inductor lowers the decomposed pad/view/slice/scatter/add/softmax/RNG/layout graph as separate generic kernels over large intermediates; Inductor cannot do this today because its pattern library does not recognize the structured Longformer band assembly feeding a stochastic softmax epilogue with live side outputs and a padded destination layout; the fix is NEW_PATTERN: add a Longformer sliding-window attention lowering that canonicalizes the band/scatter assembly, materializes required side outputs, and fuses the reduction, dropout, and final layout scatter."""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


BATCH = 8
SEQ = 1024
HEADS = 12
CHUNK = 256
CHUNKS = 4
R = 513
PADDED_R = 770
FINAL_R = 769
OUT_M = BATCH * HEADS * CHUNKS
OUT_D = 768
OUT_T = 256
ROWS = BATCH * SEQ * HEADS
BLOCK = 1024
DROP_P = 0.10009765625
DROP_SCALE = 1.1111111111111112
SEED_INDEX = 24

LOGITS_STRIDE = (HEADS * SEQ * R, R, SEQ * R, 1)
REDUCE_STRIDE = (SEQ * HEADS, HEADS, 1, 1)
FINAL_STRIDE = (OUT_T * PADDED_R, FINAL_R, 1)
FINAL_STORAGE = (OUT_M - 1) * FINAL_STRIDE[0] + (OUT_T - 1) * FINAL_STRIDE[1] + (OUT_D - 1) + 1


@triton.jit
def _round_bf16_to_f32(x):
    return tl.inline_asm_elementwise(
        "{ .reg .b16 t; cvt.rn.bf16.f32 t, $1; cvt.f32.bf16 $0, t; }",
        constraints="=f,f",
        args=[x],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _zero_bf16_storage(out_ptr, n_elements: tl.constexpr, BLOCK_N: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    tl.store(out_ptr + offsets, tl.zeros((BLOCK_N,), tl.float32), mask=offsets < n_elements)


@triton.jit
def _load_skewed_bmm(
    bmm_ptr,
    bh,
    chunk,
    skew_row,
    skew_col,
    mask,
    R_: tl.constexpr,
):
    linear = skew_row * R_ + skew_col
    src_row = linear // 512
    src_col = linear - src_row * 512
    valid = mask & (src_row < 512)
    safe_chunk = tl.where(valid, chunk, 0)
    safe_row = tl.where(valid, src_row, 0)
    safe_col = tl.where(valid, src_col, 0)
    offset = (bh * 3 + safe_chunk) * (512 * 512) + safe_row * 512 + safe_col
    return tl.load(bmm_ptr + offset, mask=valid, other=0.0).to(tl.float32)


@triton.jit
def _longformer_scope_kernel(
    bmm_ptr,
    first_base_ptr,
    first_mask_ptr,
    first_last_value_ptr,
    last_mask_ptr,
    global_bias_ptr,
    query_mask_ptr,
    fill_ptr,
    rng_or_seed_ptr,
    logits_out_ptr,
    amax_out_ptr,
    sum_out_ptr,
    keep_out_ptr,
    final_out_ptr,
    BLOCK_N: tl.constexpr,
    R_: tl.constexpr,
    SEQ_: tl.constexpr,
    HEADS_: tl.constexpr,
    CHUNK_: tl.constexpr,
    CHUNKS_: tl.constexpr,
    OUT_D_: tl.constexpr,
    PADDED_R_: tl.constexpr,
    FINAL_R_: tl.constexpr,
    LOGITS_S0: tl.constexpr,
    LOGITS_S1: tl.constexpr,
    LOGITS_S2: tl.constexpr,
    FIRST_BASE_S0: tl.constexpr,
    FINAL_S0: tl.constexpr,
    FINAL_S1: tl.constexpr,
    DROP_P_: tl.constexpr,
    DROP_SCALE_: tl.constexpr,
    SEED_INDEX_: tl.constexpr,
    USE_RANDOM_PTR: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    valid_cols = cols < R_

    head = row % HEADS_
    row_div_heads = row // HEADS_
    seq = row_div_heads % SEQ_
    batch = row_div_heads // SEQ_
    bh = batch * HEADS_ + head
    chunk_id = seq // CHUNK_
    pos = seq - chunk_id * CHUNK_

    col_i32 = cols.to(tl.int32)
    from_right = col_i32 >= CHUNK_
    source_chunk_right = tl.minimum(chunk_id, 2)
    skew_row_right = tl.where(chunk_id == 3, pos + CHUNK_, pos)
    source_chunk_left = tl.where(chunk_id == 0, 0, chunk_id - 1)
    skew_row_left = tl.where(chunk_id == 0, pos - 1, pos + CHUNK_ - 1)
    source_chunk = tl.where(from_right, source_chunk_right, source_chunk_left)
    skew_row = tl.where(from_right, skew_row_right, skew_row_left)
    skew_col = tl.where(from_right, col_i32 - CHUNK_, col_i32 + CHUNK_ + 1)
    first_left_interior = (chunk_id == 0) & (pos > 0) & (col_i32 > 0) & (col_i32 < CHUNK_)
    has_bmm_source = from_right | (chunk_id != 0) | first_left_interior

    bmm_score = _load_skewed_bmm(
        bmm_ptr,
        bh,
        source_chunk,
        skew_row,
        skew_col,
        valid_cols & has_bmm_source,
        R_,
    )

    safe_col = tl.minimum(col_i32, R_ - 1)
    first_chunk_base = tl.load(
        first_base_ptr + bh * FIRST_BASE_S0 + pos * R_ + safe_col,
        mask=valid_cols,
        other=0.0,
    ).to(tl.float32)
    first_select = (chunk_id == 0) & ((pos == 0) | (col_i32 == 0)) & (col_i32 < CHUNK_)
    local_score = tl.where(first_select, first_chunk_base, bmm_score)

    edge_col = tl.minimum(col_i32, CHUNK_)
    first_offsets = ((batch * CHUNK_ + pos) * HEADS_ + head) * (CHUNK_ + 1) + edge_col
    first_mask = tl.load(
        first_mask_ptr + first_offsets,
        mask=(chunk_id == 0) & (col_i32 < (CHUNK_ + 1)),
        other=0,
    ) != 0
    first_value = tl.load(
        first_last_value_ptr + batch * (CHUNK_ * HEADS_ * (CHUNK_ + 1)) + pos * (CHUNK_ + 1) + head * (CHUNK_ * (CHUNK_ + 1)) + edge_col,
        mask=(chunk_id == 0) & (col_i32 < (CHUNK_ + 1)),
        other=0.0,
    ).to(tl.float32)
    local_score = tl.where((chunk_id == 0) & (col_i32 < (CHUNK_ + 1)) & first_mask, first_value, local_score)

    last_col = col_i32 - CHUNK_
    safe_last_col = tl.maximum(tl.minimum(last_col, CHUNK_), 0)
    last_offsets = ((batch * CHUNK_ + pos) * HEADS_ + head) * (CHUNK_ + 1) + safe_last_col
    last_mask = tl.load(
        last_mask_ptr + last_offsets,
        mask=(chunk_id == (CHUNKS_ - 1)) & (col_i32 >= CHUNK_) & valid_cols,
        other=0,
    ) != 0
    last_value = tl.load(
        first_last_value_ptr + batch * (CHUNK_ * HEADS_ * (CHUNK_ + 1)) + pos * (CHUNK_ + 1) + head * (CHUNK_ * (CHUNK_ + 1)) + safe_last_col,
        mask=(chunk_id == (CHUNKS_ - 1)) & (col_i32 >= CHUNK_) & valid_cols,
        other=0.0,
    ).to(tl.float32)
    local_score = tl.where((chunk_id == (CHUNKS_ - 1)) & (col_i32 >= CHUNK_) & last_mask, last_value, local_score)

    bias = tl.load(
        global_bias_ptr + (batch * SEQ_ + seq) * R_ + safe_col,
        mask=valid_cols,
        other=0.0,
    ).to(tl.float32)
    scores = _round_bf16_to_f32(local_score + bias)

    logits_offsets = batch * LOGITS_S0 + seq * LOGITS_S1 + head * LOGITS_S2 + cols
    tl.store(logits_out_ptr + logits_offsets, scores, mask=valid_cols)

    nan_count = tl.sum(tl.where(valid_cols & (scores != scores), 1, 0), axis=0)
    row_max_raw = tl.max(tl.where(valid_cols, scores, -float("inf")), axis=0)
    row_max = tl.where(nan_count != 0, float("nan"), row_max_raw)
    numer = libdevice.exp(scores - row_max)
    numer = tl.where(valid_cols, numer, 0.0)
    denom = tl.sum(numer, axis=0)
    tl.store(amax_out_ptr + row, row_max)
    tl.store(sum_out_ptr + row, denom)

    values = numer / denom
    query_masked = tl.load(query_mask_ptr + batch * SEQ_ + seq) != 0
    fill = tl.load(fill_ptr).to(tl.float32)
    values = tl.where(query_masked, fill, values)
    values = _round_bf16_to_f32(values)

    if USE_RANDOM_PTR:
        random_value = tl.load(rng_or_seed_ptr + row * R_ + safe_col, mask=valid_cols, other=0.0).to(tl.float32)
    else:
        seed = tl.load(rng_or_seed_ptr + SEED_INDEX_)
        random_value = tl.rand(seed, (row * R_ + safe_col).to(tl.uint32))
    keep = _round_bf16_to_f32(random_value) > DROP_P_
    tl.store(keep_out_ptr + row * R_ + cols, keep, mask=valid_cols)

    dropped = values * keep.to(tl.float32)
    final_value = dropped * DROP_SCALE_
    padded_linear = pos * PADDED_R_ + safe_col
    out_t = padded_linear // FINAL_R_
    out_d = padded_linear - out_t * FINAL_R_
    out_m = bh * CHUNKS_ + chunk_id
    final_offsets = out_m * FINAL_S0 + out_t * FINAL_S1 + out_d
    tl.store(final_out_ptr + final_offsets, final_value, mask=valid_cols & (out_d < OUT_D_))


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


@oracle_impl(
    hardware="B200",
    point="b64f0e8a",
    num_warps=4,
)
def oracle_forward(inputs, *, num_warps: int):
    (
        bmm,
        _copy_layout,
        first_base,
        _scatter_base,
        first_mask,
        first_last_value,
        last_mask,
        global_bias,
        query_mask,
        fill,
        seeds,
        *_shape_params,
    ) = inputs

    device = bmm.device
    logits = torch.empty_strided(
        (BATCH, SEQ, HEADS, R),
        LOGITS_STRIDE,
        device=device,
        dtype=torch.bfloat16,
    )
    amax = torch.empty_strided(
        (BATCH, SEQ, HEADS, 1),
        REDUCE_STRIDE,
        device=device,
        dtype=torch.float32,
    )
    denom = torch.empty_strided(
        (BATCH, SEQ, HEADS, 1),
        REDUCE_STRIDE,
        device=device,
        dtype=torch.float32,
    )
    keep = torch.empty_strided(
        (BATCH, SEQ, HEADS, R),
        (SEQ * HEADS * R, HEADS * R, R, 1),
        device=device,
        dtype=torch.bool,
    )
    final = torch.empty_strided(
        (OUT_M, OUT_T, OUT_D),
        FINAL_STRIDE,
        device=device,
        dtype=torch.bfloat16,
    )
    _zero_bf16_storage[(triton.cdiv(FINAL_STORAGE, BLOCK),)](
        final,
        FINAL_STORAGE,
        BLOCK_N=BLOCK,
        num_warps=4,
    )
    if torch.cuda.is_current_stream_capturing():
        rng_or_seed = seeds
        use_random_ptr = False
    else:
        seed = torch.ops.prims.inductor_lookup_seed.default(seeds, SEED_INDEX)
        rng_or_seed = _inductor_random_for_eager_check(
            (BATCH, SEQ, HEADS, R),
            seed,
            device=device,
        )
        use_random_ptr = True
    _longformer_scope_kernel[(ROWS,)](
        bmm,
        first_base,
        first_mask,
        first_last_value,
        last_mask,
        global_bias,
        query_mask,
        fill,
        rng_or_seed,
        logits,
        amax,
        denom,
        keep,
        final,
        BLOCK_N=BLOCK,
        R_=R,
        SEQ_=SEQ,
        HEADS_=HEADS,
        CHUNK_=CHUNK,
        CHUNKS_=CHUNKS,
        OUT_D_=OUT_D,
        PADDED_R_=PADDED_R,
        FINAL_R_=FINAL_R,
        LOGITS_S0=LOGITS_STRIDE[0],
        LOGITS_S1=LOGITS_STRIDE[1],
        LOGITS_S2=LOGITS_STRIDE[2],
        FIRST_BASE_S0=int(first_base.stride(0)),
        FINAL_S0=FINAL_STRIDE[0],
        FINAL_S1=FINAL_STRIDE[1],
        DROP_P_=DROP_P,
        DROP_SCALE_=DROP_SCALE,
        SEED_INDEX_=SEED_INDEX,
        USE_RANDOM_PTR=use_random_ptr,
        num_warps=num_warps,
    )
    return logits, amax, denom, keep, final, final.permute(0, 2, 1)
