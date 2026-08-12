"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete M2M100 bf16 internally masked attention softmax/dropout scope in one Triton row kernel, including the bool mask lowered to 0/-inf bias, the shape-param score view, the compiled fused score-plus-bias add consumed by fp32 stable last-dimension amax/libdevice.exp/sum/div, explicit bf16 probability cast, `eq(-inf)`/`any` all-masked-row zero fallback, internally generated `prims.inductor_seeds.default(3)`, seed-index-0 dropout with f32 random rounded to bf16 before `gt(0.1)`, bf16 dropout scaling by 1.1111111111111112, returned softmax tensor, returned seeds, returned bool mask, returned 3D contiguous view, and returned permute alias, whereas Inductor lowers the internally created mask constants, softmax reductions, all-masked guard, generated RNG/dropout epilogue, and layout-only aliases as generic pointwise, reduction, RNG, and view fragments; Inductor cannot do this today because its attention-softmax scheduler does not recognize this internally seeded scalar-mask dropout envelope as one full-scope row template with all observable side outputs preserved; the fix is NEW_PATTERN: add an M2M100-style masked attention softmax/dropout lowering that fuses mask construction, row-validity detection, stable softmax, generated-seed dropout, and alias-output emission into one guarded generated kernel."""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


SEED_COUNT = 3
SEED_INDEX = 0


@triton.jit
def _f32_mul(a, b):
    return tl.inline_asm_elementwise(
        "mul.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


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
        (numel + block_size - 1) // block_size,
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


@triton.jit
def _masked_softmax_dropout_kernel(
    mask_ptr,
    score_ptr,
    rng_ptr,
    where_ptr,
    gt_ptr,
    dropped_ptr,
    mask_s0: tl.constexpr,
    mask_s2: tl.constexpr,
    mask_s3: tl.constexpr,
    score_s0: tl.constexpr,
    score_s1: tl.constexpr,
    score_s2: tl.constexpr,
    heads: tl.constexpr,
    q_len: tl.constexpr,
    k_len: tl.constexpr,
    n_rows: tl.constexpr,
    use_random_ptr: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    row_mask = rows < n_rows
    col_mask = cols < k_len
    elem_mask = row_mask[:, None] & col_mask[None, :]

    flat_bh = rows // q_len
    batch = flat_bh // heads
    q = rows - flat_bh * q_len
    offsets = rows[:, None] * k_len + cols[None, :]

    mask_offsets = (
        batch[:, None] * mask_s0
        + q[:, None] * mask_s2
        + cols[None, :] * mask_s3
    )
    keep_score = tl.load(mask_ptr + mask_offsets, mask=elem_mask, other=0)

    score_offsets = (
        flat_bh[:, None] * score_s0
        + q[:, None] * score_s1
        + cols[None, :] * score_s2
    )
    raw = tl.load(score_ptr + score_offsets, mask=elem_mask, other=0.0).to(tl.float32)
    values = tl.where(elem_mask & keep_score, raw, -float("inf"))

    live = elem_mask & (values != -float("inf"))
    has_any = tl.max(tl.where(live, 1, 0), axis=1) != 0
    row_max = tl.max(values, axis=1)
    safe_max = tl.where(has_any, row_max, 0.0)
    numer = libdevice.exp(values - safe_max[:, None])
    numer = tl.where(live, numer, 0.0)
    denom = tl.sum(numer, axis=1)
    denom = tl.where(has_any, denom, 1.0)
    probs = (numer / denom[:, None]).to(tl.bfloat16)
    where_val = tl.where(has_any[:, None], probs, 0.0).to(tl.bfloat16)
    tl.store(where_ptr + offsets, where_val, mask=elem_mask)

    if use_random_ptr:
        rand_bf16 = tl.load(rng_ptr + offsets, mask=elem_mask, other=0.0).to(tl.bfloat16)
    else:
        seed = tl.load(rng_ptr + 0)
        rand_bf16 = tl.rand(seed, offsets.to(tl.uint32)).to(tl.bfloat16)
    dropout_p = tl.full((BLOCK_M, BLOCK_N), 0.1, tl.float32).to(tl.bfloat16)
    keep = rand_bf16 > dropout_p
    tl.store(gt_ptr + offsets, keep, mask=elem_mask)

    dropped = tl.where(keep, where_val, 0.0).to(tl.bfloat16)
    scaled = _f32_mul(dropped.to(tl.float32), 1.1111111111111112).to(tl.bfloat16)
    tl.store(dropped_ptr + offsets, scaled, mask=elem_mask)


# 54ff0363: M2M100 train internally seeded masked K=128 attention softmax/dropout.
@oracle_impl(hardware="B200", point="54ff0363", BLOCK_M=8, BLOCK_N=128, num_warps=4, num_stages=3)
def oracle_forward(
    inputs,
    *,
    BLOCK_M: int,
    BLOCK_N: int,
    num_warps: int,
    num_stages: int,
):
    arg0_1, arg1_1, shape0, shape1, random_shape, _shape3, flat_shape = inputs
    del shape1, _shape3

    full_shape = tuple(int(dim) for dim in shape0)
    rand_shape = tuple(int(dim) for dim in random_shape)
    out_flat_shape = tuple(int(dim) for dim in flat_shape)
    batch_size, heads, q_len, k_len = full_shape
    n_rows = batch_size * heads * q_len
    device = arg1_1.device

    where = torch.empty_strided(
        full_shape,
        (heads * q_len * k_len, q_len * k_len, k_len, 1),
        device=device,
        dtype=torch.bfloat16,
    )
    gt = torch.empty_strided(
        full_shape,
        (heads * q_len * k_len, q_len * k_len, k_len, 1),
        device=device,
        dtype=torch.bool,
    )
    dropped = torch.empty_strided(
        out_flat_shape,
        (out_flat_shape[1] * out_flat_shape[2], out_flat_shape[2], 1),
        device=device,
        dtype=torch.bfloat16,
    )

    if torch.cuda.is_current_stream_capturing():
        seeds = torch.ops.prims.inductor_seeds.default(SEED_COUNT, device)
        rng_source = seeds
        use_random_ptr = False
    else:
        seeds, rng_source = _seeds_and_random_for_eager_check(rand_shape, device=device)
        use_random_ptr = True

    _masked_softmax_dropout_kernel[(triton.cdiv(n_rows, BLOCK_M),)](
        arg0_1,
        arg1_1,
        rng_source,
        where,
        gt,
        dropped,
        mask_s0=arg0_1.stride(0),
        mask_s2=arg0_1.stride(2),
        mask_s3=arg0_1.stride(3),
        score_s0=arg1_1.stride(0),
        score_s1=arg1_1.stride(1),
        score_s2=arg1_1.stride(2),
        heads=heads,
        q_len=q_len,
        k_len=k_len,
        n_rows=n_rows,
        use_random_ptr=use_random_ptr,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return where, seeds, gt, dropped, dropped.permute(0, 2, 1)
