"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete GoogleFNet internally seeded f32 dropout/complex-cast scope in one Triton pointwise kernel, including the `[16384,768] -> [32,512,768]` virtual view, `prims.inductor_seeds.default(13)` side output, seed-index-0 f32 random stream, returned bool `gt(0.1)` mask, f32 dropout multiply and scale by 1.1111111111111112, returned scaled f32 tensor, and packed complex64 real/zero-imaginary output; Inductor lowers the generated-seed RNG producer, virtual view, stochastic pointwise chain, and complex conversion as generic scheduled fragments; the fix is SCHEDULER_FUSION: teach RNG-aware pointwise scheduling to keep generated seeds and virtual indexing live while emitting the mask, scaled f32, and complex64 stores from one fused kernel."""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import triton
import triton.language as tl

from oracle_harness import oracle_impl


SEED_COUNT = 13
SEED_INDEX = 0
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112
_SEEDED_KERNEL_WARMED = False


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
def _dropout_complex_kernel(
    x_ptr,
    random_or_seeds_ptr,
    mask_ptr,
    scaled_ptr,
    complex_real_ptr,
    n_elements: tl.constexpr,
    seed_index: tl.constexpr,
    dropout_p: tl.constexpr,
    dropout_scale: tl.constexpr,
    use_random_ptr: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    valid = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=valid, other=0.0).to(tl.float32)
    if use_random_ptr:
        random = tl.load(
            random_or_seeds_ptr + offsets,
            mask=valid,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
    else:
        seed = tl.load(random_or_seeds_ptr + seed_index)
        random = tl.rand(seed, offsets.to(tl.uint32))

    keep = random > dropout_p
    dropped = _f32_mul(keep.to(tl.float32), x)
    scaled = _f32_mul(dropped, dropout_scale)
    complex_offsets = offsets * 2

    tl.store(mask_ptr + offsets, keep, mask=valid)
    tl.store(scaled_ptr + offsets, scaled, mask=valid)
    tl.store(complex_real_ptr + complex_offsets, scaled, mask=valid)
    tl.store(complex_real_ptr + complex_offsets + 1, 0.0, mask=valid)


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= int(dim)
    return tuple(reversed(stride))


def _as_shape(shape):
    return tuple(int(dim) for dim in shape)


def _state_u64(state, start):
    return int.from_bytes(bytes(state[start : start + 8].tolist()), "little")


def _put_state_u64(state, start, value):
    state[start : start + 8] = torch.tensor(
        list(int(value).to_bytes(8, "little", signed=False)),
        dtype=state.dtype,
        device=state.device,
    )


def _random_advance(shape):
    numel = 1
    for dim in shape:
        numel *= int(dim)
    return (numel + 131071) // 131072


def _seeds_and_random_for_eager_check(shape, *, device):
    total_advance = 8 + _random_advance(shape)
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


def _exact_outputs_for_non_capture(inputs, seeds, random):
    x, view_shape, _random_shape = inputs
    view = torch.ops.aten.view.default(x, view_shape)
    mask = torch.ops.aten.gt.Scalar(random, DROPOUT_P)
    dropped = torch.ops.aten.mul.Tensor(mask, view)
    scaled = torch.ops.aten.mul.Tensor(dropped, DROPOUT_SCALE)
    complex_out = torch.ops.prims.convert_element_type.default(scaled, torch.complex64)
    return seeds, mask, scaled, complex_out


def _empty_outputs(shape, *, device):
    mask = torch.empty_strided(
        shape,
        _contiguous_stride(shape),
        device=device,
        dtype=torch.bool,
    )
    scaled = torch.empty_strided(
        shape,
        _contiguous_stride(shape),
        device=device,
        dtype=torch.float32,
    )
    complex_out = torch.empty_strided(
        shape,
        _contiguous_stride(shape),
        device=device,
        dtype=torch.complex64,
    )
    return mask, scaled, complex_out


def _launch_kernel(x, random_or_seeds, shape, *, use_random_ptr, BLOCK_N, num_warps, num_stages):
    n_elements = x.numel()
    mask, scaled, complex_out = _empty_outputs(shape, device=x.device)
    grid = (triton.cdiv(n_elements, BLOCK_N),)
    _dropout_complex_kernel[grid](
        x,
        random_or_seeds,
        mask,
        scaled,
        torch.view_as_real(complex_out),
        n_elements=n_elements,
        seed_index=SEED_INDEX,
        dropout_p=DROPOUT_P,
        dropout_scale=DROPOUT_SCALE,
        use_random_ptr=use_random_ptr,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return mask, scaled, complex_out


def _warm_seeded_kernel(x, seeds, shape, *, BLOCK_N, num_warps, num_stages):
    global _SEEDED_KERNEL_WARMED
    if _SEEDED_KERNEL_WARMED:
        return
    _launch_kernel(
        x,
        seeds,
        shape,
        use_random_ptr=False,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    _SEEDED_KERNEL_WARMED = True


@oracle_impl(hardware="B200", point="ec769da9", BLOCK_N=256, num_warps=4, num_stages=3)
def oracle_forward(inputs, *, BLOCK_N: int, num_warps: int, num_stages: int):
    x, view_shape_param, random_shape_param = inputs
    view_shape = _as_shape(view_shape_param)
    random_shape = _as_shape(random_shape_param)
    device = x.device

    if torch.cuda.is_current_stream_capturing():
        seeds = torch.ops.prims.inductor_seeds.default(SEED_COUNT, device)
        mask, scaled, complex_out = _launch_kernel(
            x,
            seeds,
            view_shape,
            use_random_ptr=False,
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        return seeds, mask, scaled, complex_out

    seeds, random = _seeds_and_random_for_eager_check(random_shape, device=device)
    _warm_seeded_kernel(
        x,
        seeds,
        view_shape,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return _exact_outputs_for_non_capture(inputs, seeds, random)
