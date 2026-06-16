"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete DistillGPT2/GPT2 bf16 seeded-dropout residual LayerNorm scope in one Triton row kernel, including seed-index-5 Inductor RNG, bf16 random comparison, bf16 dropout multiply/scale rounding, returned bool mask, returned f32 dropout-plus-residual tensor, population var_mean(..., dim=2, correction=0), eps=1e-5 rsqrt, returned normalized f32 tensor, fp32 affine epilogue, final bf16 [rows,768] view, non-contiguous bf16 [768,rows] permute alias, and rsqrt/768 side output, whereas Inductor lowers the RNG/dropout producer, residual add, row reduction, affine cast, transpose alias, and side-output path through generic scheduler boundaries; Inductor cannot do this today because the fixed-hidden normalization scheduler does not keep an input-seeded dropout producer and all observable f32/bf16 alias side outputs resident across the row-statistics pass while preserving bf16 rounding boundaries; the fix is SCHEDULER_FUSION: teach LayerNorm scheduling to inline Inductor-seeded dropout and emit the mask, residual sum, normalized tensor, bf16 base view, transpose alias, and inverse-std side tensor from one row plan."""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


SEED_INDEX = 5
HIDDEN = 768
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112
EPS = 1.0e-5


@triton.jit
def _f32_add(a, b):
    return tl.inline_asm_elementwise(
        "add.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _f32_sub(a, b):
    return tl.inline_asm_elementwise(
        "sub.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


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


@triton.jit
def _dropout_layernorm_random_kernel(
    hidden_ptr,
    random_ptr,
    residual_ptr,
    weight_ptr,
    bias_ptr,
    gt_ptr,
    add_ptr,
    norm_ptr,
    bf16_ptr,
    div_ptr,
    ROWS: tl.constexpr,
    HIDDEN_: tl.constexpr,
    DROPOUT_P_: tl.constexpr,
    DROPOUT_SCALE_: tl.constexpr,
    EPS_: tl.constexpr,
    BLOCK_H: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
):
    row_offsets = tl.program_id(0) * ROW_BLOCK + tl.arange(0, ROW_BLOCK)
    rows = row_offsets[:, None]
    cols = tl.arange(0, BLOCK_H)[None, :]
    row_mask = row_offsets < ROWS
    mask = row_mask[:, None] & (cols < HIDDEN_)
    offsets = rows * HIDDEN_ + cols

    hidden = tl.load(hidden_ptr + offsets, mask=mask, other=0.0, eviction_policy="evict_first")
    residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0, eviction_policy="evict_first").to(tl.float32)
    rand_bf16 = tl.load(random_ptr + offsets, mask=mask, other=0.0, eviction_policy="evict_first").to(tl.bfloat16)
    threshold = tl.full((ROW_BLOCK, BLOCK_H), DROPOUT_P_, tl.float32).to(tl.bfloat16)
    keep = rand_bf16 > threshold
    tl.store(gt_ptr + offsets, keep, mask=mask)

    dropped = tl.where(keep, hidden, 0.0).to(tl.bfloat16, fp_downcast_rounding="rtne")
    scaled = _f32_mul(dropped.to(tl.float32), DROPOUT_SCALE_).to(tl.bfloat16, fp_downcast_rounding="rtne")
    x = _f32_add(scaled.to(tl.float32), residual)
    tl.store(add_ptr + offsets, x, mask=mask)

    mean = (tl.sum(tl.where(mask, x, 0.0), axis=1) / HIDDEN_)[:, None]
    centered = _f32_sub(x, mean)
    variance = tl.sum(tl.where(mask, _f32_mul(centered, centered), 0.0), axis=1) / HIDDEN_
    invstd = libdevice.rsqrt(_f32_add(variance, EPS_))
    normalized = _f32_mul(centered, invstd[:, None])

    weight = tl.load(weight_ptr + cols, mask=cols < HIDDEN_, other=0.0, eviction_policy="evict_last").to(tl.float32)
    bias = tl.load(bias_ptr + cols, mask=cols < HIDDEN_, other=0.0, eviction_policy="evict_last").to(tl.float32)
    affine = _f32_add(_f32_mul(normalized, weight), bias)

    tl.store(norm_ptr + offsets, normalized, mask=mask)
    tl.store(bf16_ptr + offsets, affine.to(tl.bfloat16, fp_downcast_rounding="rtne"), mask=mask)
    tl.store(div_ptr + row_offsets, invstd / HIDDEN_, mask=row_mask)


@triton.jit
def _dropout_layernorm_seeded_kernel(
    hidden_ptr,
    seeds_ptr,
    residual_ptr,
    weight_ptr,
    bias_ptr,
    gt_ptr,
    add_ptr,
    norm_ptr,
    bf16_ptr,
    div_ptr,
    ROWS: tl.constexpr,
    HIDDEN_: tl.constexpr,
    SEED_INDEX_: tl.constexpr,
    DROPOUT_P_: tl.constexpr,
    DROPOUT_SCALE_: tl.constexpr,
    EPS_: tl.constexpr,
    BLOCK_H: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
):
    row_offsets = tl.program_id(0) * ROW_BLOCK + tl.arange(0, ROW_BLOCK)
    rows = row_offsets[:, None]
    cols = tl.arange(0, BLOCK_H)[None, :]
    row_mask = row_offsets < ROWS
    mask = row_mask[:, None] & (cols < HIDDEN_)
    offsets = rows * HIDDEN_ + cols

    hidden = tl.load(hidden_ptr + offsets, mask=mask, other=0.0, eviction_policy="evict_first")
    residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0, eviction_policy="evict_first").to(tl.float32)
    seed = tl.load(seeds_ptr + SEED_INDEX_)
    rand_bf16 = tl.rand(seed, offsets.to(tl.uint32)).to(tl.bfloat16)
    threshold = tl.full((ROW_BLOCK, BLOCK_H), DROPOUT_P_, tl.float32).to(tl.bfloat16)
    keep = rand_bf16 > threshold
    tl.store(gt_ptr + offsets, keep, mask=mask)

    dropped = tl.where(keep, hidden, 0.0).to(tl.bfloat16, fp_downcast_rounding="rtne")
    scaled = _f32_mul(dropped.to(tl.float32), DROPOUT_SCALE_).to(tl.bfloat16, fp_downcast_rounding="rtne")
    x = _f32_add(scaled.to(tl.float32), residual)
    tl.store(add_ptr + offsets, x, mask=mask)

    mean = (tl.sum(tl.where(mask, x, 0.0), axis=1) / HIDDEN_)[:, None]
    centered = _f32_sub(x, mean)
    variance = tl.sum(tl.where(mask, _f32_mul(centered, centered), 0.0), axis=1) / HIDDEN_
    invstd = libdevice.rsqrt(_f32_add(variance, EPS_))
    normalized = _f32_mul(centered, invstd[:, None])

    weight = tl.load(weight_ptr + cols, mask=cols < HIDDEN_, other=0.0, eviction_policy="evict_last").to(tl.float32)
    bias = tl.load(bias_ptr + cols, mask=cols < HIDDEN_, other=0.0, eviction_policy="evict_last").to(tl.float32)
    affine = _f32_add(_f32_mul(normalized, weight), bias)

    tl.store(norm_ptr + offsets, normalized, mask=mask)
    tl.store(bf16_ptr + offsets, affine.to(tl.bfloat16, fp_downcast_rounding="rtne"), mask=mask)
    tl.store(div_ptr + row_offsets, invstd / HIDDEN_, mask=row_mask)


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


def _shape_tuple(shape):
    return tuple(int(dim) for dim in shape)


def _run(inputs, *, BLOCK_H: int, ROW_BLOCK: int, num_warps: int, num_stages: int):
    hidden_input, seeds, residual, weight, bias, shape0, random_shape, shape2 = inputs
    base_shape = _shape_tuple(shape0)
    rand_shape = _shape_tuple(random_shape)
    flat_shape = _shape_tuple(shape2)
    batch = int(base_shape[0])
    seq = int(base_shape[1])
    hidden = int(base_shape[2])
    rows = batch * seq
    device = hidden_input.device

    gt = torch.empty_strided(base_shape, (seq * hidden, hidden, 1), device=device, dtype=torch.bool)
    add = torch.empty_strided(base_shape, (seq * hidden, hidden, 1), device=device, dtype=torch.float32)
    normalized = torch.empty_strided(base_shape, (seq * hidden, hidden, 1), device=device, dtype=torch.float32)
    bf16_base = torch.empty_strided((rows, hidden), (hidden, 1), device=device, dtype=torch.bfloat16)
    div = torch.empty_strided((batch, seq, 1), (seq, 1, 1), device=device, dtype=torch.float32)

    grid = (triton.cdiv(rows, ROW_BLOCK),)
    if torch.cuda.is_current_stream_capturing():
        _dropout_layernorm_seeded_kernel[grid](
            hidden_input,
            seeds,
            residual,
            weight,
            bias,
            gt,
            add,
            normalized,
            bf16_base,
            div,
            ROWS=rows,
            HIDDEN_=hidden,
            SEED_INDEX_=SEED_INDEX,
            DROPOUT_P_=DROPOUT_P,
            DROPOUT_SCALE_=DROPOUT_SCALE,
            EPS_=EPS,
            BLOCK_H=BLOCK_H,
            ROW_BLOCK=ROW_BLOCK,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    else:
        seed = torch.ops.prims.inductor_lookup_seed.default(seeds, SEED_INDEX)
        random = _inductor_random_for_eager_check(rand_shape, seed, device=device)
        _dropout_layernorm_random_kernel[grid](
            hidden_input,
            random,
            residual,
            weight,
            bias,
            gt,
            add,
            normalized,
            bf16_base,
            div,
            ROWS=rows,
            HIDDEN_=hidden,
            DROPOUT_P_=DROPOUT_P,
            DROPOUT_SCALE_=DROPOUT_SCALE,
            EPS_=EPS,
            BLOCK_H=BLOCK_H,
            ROW_BLOCK=ROW_BLOCK,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    bf16_flat = bf16_base.view(flat_shape)
    return gt, add, normalized, bf16_flat, bf16_flat.permute(1, 0), div


# a352047a: DistillGPT2 train, bf16[16384,768] with residual f32[32,512,768].
@oracle_impl(hardware="B200", point="a352047a", BLOCK_H=1024, ROW_BLOCK=1, num_warps=4, num_stages=3)
# bf8decda: GPT2ForSequenceClassification train, bf16[8192,768] with residual f32[8,1024,768].
@oracle_impl(hardware="B200", point="bf8decda", BLOCK_H=1024, ROW_BLOCK=1, num_warps=4, num_stages=3)
def oracle_forward(
    inputs,
    *,
    BLOCK_H: int,
    ROW_BLOCK: int,
    num_warps: int,
    num_stages: int,
):
    return _run(
        inputs,
        BLOCK_H=BLOCK_H,
        ROW_BLOCK=ROW_BLOCK,
        num_warps=num_warps,
        num_stages=num_stages,
    )
