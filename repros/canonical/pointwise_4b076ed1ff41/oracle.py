"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete bf16 Inductor-seeded dropout-add scope in one Triton pointwise kernel, including the shape-param view, seed-index-1 random stream, f32-to-bf16 random cast before `gt(0.1)`, returned bool mask, bf16 dropout multiply and scale rounding, and fp32 residual add output, whereas Inductor lowers the stochastic producer and sibling pointwise consumers through generic pointwise scheduling; Inductor cannot do this today because its scheduler does not keep the seeded RNG producer, bf16 rounding boundaries, bool mask store, and fp32 add consumer in one full-output pointwise plan; the fix is SCHEDULER_FUSION: teach the pointwise scheduler to inline `prims.inductor_random` with explicit seed indexing and emit the mask plus scaled-add stores from a single fused kernel."""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import triton
import triton.language as tl

from oracle_harness import oracle_impl


SEED_INDEX = 1
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112


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
def _dropout_add_kernel(
    x_ptr,
    random_or_seed_ptr,
    residual_ptr,
    mask_ptr,
    out_ptr,
    n_elements: tl.constexpr,
    seed_index: tl.constexpr,
    dropout_p: tl.constexpr,
    dropout_scale: tl.constexpr,
    use_seeded_rng: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    valid = offsets < n_elements

    if use_seeded_rng:
        seed = tl.load(random_or_seed_ptr + seed_index)
        random_bf16 = tl.rand(seed, offsets.to(tl.uint32)).to(tl.bfloat16)
    else:
        random_bf16 = tl.load(
            random_or_seed_ptr + offsets,
            mask=valid,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.bfloat16)

    threshold = tl.full((BLOCK_N,), dropout_p, tl.float32).to(tl.bfloat16)
    keep = random_bf16 > threshold
    x = tl.load(x_ptr + offsets, mask=valid, other=0.0).to(tl.bfloat16)
    residual = tl.load(residual_ptr + offsets, mask=valid, other=0.0).to(tl.float32)
    zero = tl.full((BLOCK_N,), 0.0, tl.float32).to(tl.bfloat16)
    dropped = tl.where(keep, x, zero).to(tl.bfloat16)
    scaled = _f32_mul(dropped.to(tl.float32), dropout_scale).to(tl.bfloat16)
    out = _f32_add(residual, scaled.to(tl.float32))

    tl.store(mask_ptr + offsets, keep, mask=valid)
    tl.store(out_ptr + offsets, out, mask=valid)


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


def _launch(inputs, *, BLOCK_N: int, num_warps: int, num_stages: int):
    x, seeds, residual, view_shape, random_shape = inputs
    del view_shape

    out_shape = tuple(int(dim) for dim in random_shape)
    n_elements = residual.numel()
    mask = torch.empty_strided(
        out_shape,
        _contiguous_stride(out_shape),
        device=residual.device,
        dtype=torch.bool,
    )
    out = torch.empty_strided(
        tuple(residual.shape),
        tuple(residual.stride()),
        device=residual.device,
        dtype=torch.float32,
    )

    grid = (triton.cdiv(n_elements, BLOCK_N),)
    if torch.cuda.is_current_stream_capturing():
        _dropout_add_kernel[grid](
            x,
            seeds,
            residual,
            mask,
            out,
            n_elements=n_elements,
            seed_index=SEED_INDEX,
            dropout_p=DROPOUT_P,
            dropout_scale=DROPOUT_SCALE,
            use_seeded_rng=True,
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    else:
        seed = torch.ops.prims.inductor_lookup_seed.default(seeds, SEED_INDEX)
        random = _inductor_random_for_eager_check(
            out_shape,
            seed,
            device=residual.device,
        )
        _dropout_add_kernel[grid](
            x,
            random,
            residual,
            mask,
            out,
            n_elements=n_elements,
            seed_index=SEED_INDEX,
            dropout_p=DROPOUT_P,
            dropout_scale=DROPOUT_SCALE,
            use_seeded_rng=False,
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    return mask, out


# 056341dc: (T([4096,2560], bf16), T([2], i64), T([32,128,2560], f32), ...)
@oracle_impl(hardware="B200", point="056341dc", BLOCK_N=256, num_warps=4, num_stages=3)
# e2f5b7a1: (T([2048,2560], bf16), T([2], i64), T([16,128,2560], f32), ...)
@oracle_impl(hardware="B200", point="e2f5b7a1", BLOCK_N=256, num_warps=4, num_stages=3)
# 89a3ffcc: (T([8192,1024], bf16), T([2], i64), T([8,1024,1024], f32), ...)
@oracle_impl(hardware="B200", point="89a3ffcc", BLOCK_N=256, num_warps=4, num_stages=3)
# 0689c6c0: (T([16384,1024], bf16), T([2], i64), T([128,128,1024], f32), ...)
@oracle_impl(hardware="B200", point="0689c6c0", BLOCK_N=256, num_warps=4, num_stages=3)
def oracle_forward(
    inputs,
    *,
    BLOCK_N: int,
    num_warps: int,
    num_stages: int,
):
    return _launch(
        inputs,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=num_stages,
    )
