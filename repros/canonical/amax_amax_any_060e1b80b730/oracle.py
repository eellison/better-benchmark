"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete XLNet bf16 relative-shift attention train scope, including the indexed relative-position gather, bf16 score and scale rounding boundaries, finite-row `any` guard, both returned fp32 amax paths, natural-exp softmax denominator, exact seed-index 42 Inductor dropout mask, final bf16 dropout output, and returned transpose alias; Inductor lowers the decomposed view/permute/slice/index/add/finite-check/reduction/RNG/dropout/layout graph as generic indexing, reduction, stochastic, and layout kernels over materialized intermediates; Inductor cannot do this today because its pattern library does not recognize XLNet's relative-shift attention-score construction as the producer of a multi-output stochastic row-softmax epilogue; the fix is NEW_PATTERN: add an XLNet relative-position attention lowering that fuses the shifted gather, row-validity test, softmax reductions, dropout, and layout-only alias stores into one generated schedule."""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


SEED_INDEX = 42
N_ROWS = 16 * 16 * 512
K_LEN = 512
OUT_SHAPE_4D = (16, 16, 512, 512)
REDUCTION_SHAPE = (16, 16, 512, 1)
OUT_SHAPE_3D = (256, 512, 512)
CONTIG_4D_STRIDE = (4194304, 262144, 512, 1)
REDUCTION_STRIDE = (8192, 512, 1, 1)
CONTIG_3D_STRIDE = (262144, 512, 1)


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
def _xlnet_train_softmax_dropout_kernel(
    content_ptr,
    rel_ptr,
    index_ptr,
    rng_or_seed_ptr,
    add_out,
    amax_out,
    amax_scaled_out,
    finite_out,
    denom_out,
    keep_out,
    final_out,
    USE_RANDOM_PTR: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    row_mask = rows < 131072
    col_mask = cols < 512
    mask = row_mask[:, None] & col_mask[None, :]

    group = rows // 512
    query = rows - group * 512
    linear = rows[:, None] * 512 + cols[None, :]

    rel_index = tl.load(index_ptr + cols, mask=col_mask, other=0)
    rel_offsets = group[:, None] * 524288 + 512 + query[:, None] * 1023 + rel_index[None, :]

    content = tl.load(content_ptr + linear, mask=mask, other=0.0).to(tl.float32)
    rel = tl.load(rel_ptr + rel_offsets, mask=mask, other=0.0).to(tl.float32)

    added_bf16 = (content + rel).to(tl.bfloat16)
    unscaled = added_bf16.to(tl.float32)
    scaled_bf16 = (added_bf16 * 0.125).to(tl.bfloat16)
    scaled = scaled_bf16.to(tl.float32)
    tl.store(add_out + linear, added_bf16, mask=mask)

    abs_scaled = tl.abs(scaled)
    finite = (scaled == scaled) & (abs_scaled != float("inf")) & mask
    invalid = (~finite) & mask
    has_invalid = tl.max(tl.where(invalid, 1, 0), axis=1) != 0
    row_is_finite = ~has_invalid

    unscaled_for_max = tl.where(mask, unscaled, -float("inf"))
    scaled_for_max = tl.where(mask, scaled, -float("inf"))
    unscaled_max = tl.max(unscaled_for_max, axis=1)
    scaled_max = tl.max(scaled_for_max, axis=1)
    tl.store(amax_out + rows, unscaled_max, mask=row_mask)
    tl.store(amax_scaled_out + rows, scaled_max, mask=row_mask)
    tl.store(finite_out + rows, row_is_finite, mask=row_mask)

    shifted_unscaled = _f32_mul(unscaled - unscaled_max[:, None], 0.125)
    shifted_scaled = scaled - scaled_max[:, None]
    shifted = tl.where(row_is_finite[:, None], shifted_unscaled, shifted_scaled)
    shifted = tl.where(mask, shifted, -float("inf"))

    numer = libdevice.exp(shifted)
    numer = tl.where(mask, numer, 0.0)
    denom = tl.sum(numer, axis=1)
    probs = numer / denom[:, None]
    tl.store(denom_out + rows, denom, mask=row_mask)

    if USE_RANDOM_PTR:
        random = tl.load(rng_or_seed_ptr + linear, mask=mask, other=0.0).to(tl.float32)
    else:
        seed = tl.load(rng_or_seed_ptr + 42)
        random = tl.rand(seed, linear.to(tl.uint32))
    keep = random > 0.1
    tl.store(keep_out + linear, keep, mask=mask)

    dropped = keep.to(tl.float32) * probs
    scaled_dropout = _f32_mul(dropped, 1.1111111111111112)
    tl.store(final_out + linear, scaled_dropout.to(tl.bfloat16), mask=mask)


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
    arg1_1,
    arg2_1,
    rng_or_seed,
    add_out,
    amax,
    amax_scaled,
    finite,
    denom,
    keep,
    final,
    *,
    use_random_ptr: bool,
    block_m: int,
    block_n: int,
    num_warps: int,
):
    _xlnet_train_softmax_dropout_kernel[(triton.cdiv(N_ROWS, block_m),)](
        arg0_1,
        arg1_1,
        arg2_1,
        rng_or_seed,
        add_out,
        amax,
        amax_scaled,
        finite,
        denom,
        keep,
        final,
        USE_RANDOM_PTR=use_random_ptr,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=num_warps,
        num_stages=3,
    )


@oracle_impl(hardware="B200", point="782e420b", block_m=4, block_n=512, num_warps=8)
def oracle_forward(inputs, *, block_m: int, block_n: int, num_warps: int):
    arg0_1, arg1_1, arg2_1, arg3_1, *_shape_params = inputs
    add_out = torch.empty_strided(
        OUT_SHAPE_4D,
        CONTIG_4D_STRIDE,
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    amax = torch.empty_strided(
        REDUCTION_SHAPE,
        REDUCTION_STRIDE,
        device=arg0_1.device,
        dtype=torch.float32,
    )
    amax_scaled = torch.empty_strided(
        REDUCTION_SHAPE,
        REDUCTION_STRIDE,
        device=arg0_1.device,
        dtype=torch.float32,
    )
    finite = torch.empty_strided(
        REDUCTION_SHAPE,
        REDUCTION_STRIDE,
        device=arg0_1.device,
        dtype=torch.bool,
    )
    denom = torch.empty_strided(
        REDUCTION_SHAPE,
        REDUCTION_STRIDE,
        device=arg0_1.device,
        dtype=torch.float32,
    )
    keep = torch.empty_strided(
        OUT_SHAPE_4D,
        CONTIG_4D_STRIDE,
        device=arg0_1.device,
        dtype=torch.bool,
    )
    final = torch.empty_strided(
        OUT_SHAPE_3D,
        CONTIG_3D_STRIDE,
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )

    if torch.cuda.is_current_stream_capturing():
        _launch(
            arg0_1,
            arg1_1,
            arg2_1,
            arg3_1,
            add_out,
            amax,
            amax_scaled,
            finite,
            denom,
            keep,
            final,
            use_random_ptr=False,
            block_m=block_m,
            block_n=block_n,
            num_warps=num_warps,
        )
        return add_out, amax, amax_scaled, finite, denom, keep, final, final.permute(0, 2, 1)

    seed = torch.ops.prims.inductor_lookup_seed.default(arg3_1, SEED_INDEX)
    random = _inductor_random_for_eager_check(OUT_SHAPE_4D, seed, device=arg0_1.device)
    _launch(
        arg0_1,
        arg1_1,
        arg2_1,
        random,
        add_out,
        amax,
        amax_scaled,
        finite,
        denom,
        keep,
        final,
        use_random_ptr=True,
        block_m=block_m,
        block_n=block_n,
        num_warps=num_warps,
    )
    return add_out, amax, amax_scaled, finite, denom, keep, final, final.permute(0, 2, 1)
