"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete bf16 MegatronBERT dropout-residual-LayerNorm scope in one hidden-size-1024 Triton row kernel, including the seed-index-12 f32-random-to-bf16 dropout mask, bf16 dropout scaling, residual add, population var_mean over the last dimension, normalized tensor, bf16 affine output view, and rsqrt/1024 side output, whereas Inductor lowers the observable mask, pre-normalization add, normalization output, bf16 affine store, and side-output work through generic norm-template scheduling; Inductor cannot do this today because normalization-template fusion does not keep stochastic bf16 dropout producers and multiple live full-row side outputs resident inside one row-reduction epilogue; the fix is SCHEDULER_FUSION: extend the norm-template scheduler to fuse seeded bf16 dropout, residual statistics, observable intermediates, affine conversion, and inverse-std side stores into one full-scope row plan."""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import triton
import triton.language as tl

from oracle_harness import oracle_impl


BATCH = 16
SEQ_LEN = 512
HIDDEN = 1024
N_ROWS = BATCH * SEQ_LEN
SEED_INDEX = 12
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112
EPS = 1.0e-12
BLOCK_H = 1024

_SEEDED_VARIANT_READY = False


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


@triton.autotune(
    configs=[
        triton.Config({"ROW_BLOCK": 1}, num_warps=4, num_stages=3),
        triton.Config({"ROW_BLOCK": 1}, num_warps=8, num_stages=3),
        triton.Config({"ROW_BLOCK": 2}, num_warps=4, num_stages=3),
        triton.Config({"ROW_BLOCK": 2}, num_warps=8, num_stages=3),
        triton.Config({"ROW_BLOCK": 4}, num_warps=8, num_stages=3),
    ],
    key=["use_random_ptr"],
)
@triton.jit
def _dropout_residual_layernorm_kernel(
    projected_ptr,
    random_or_seed_ptr,
    residual_ptr,
    weight_ptr,
    bias_ptr,
    gt_ptr,
    add_ptr,
    norm_ptr,
    out_ptr,
    invstd_div_ptr,
    total_rows: tl.constexpr,
    hidden: tl.constexpr,
    seed_index: tl.constexpr,
    dropout_p: tl.constexpr,
    dropout_scale: tl.constexpr,
    eps: tl.constexpr,
    use_random_ptr: tl.constexpr,
    BLOCK_H: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
):
    row_ids = tl.program_id(0) * ROW_BLOCK + tl.arange(0, ROW_BLOCK)
    cols = tl.arange(0, BLOCK_H)
    mask = (row_ids[:, None] < total_rows) & (cols[None, :] < hidden)
    col_mask = cols < hidden
    row_mask = row_ids < total_rows
    offsets = row_ids[:, None] * hidden + cols[None, :]

    projected = tl.load(
        projected_ptr + offsets,
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    )

    if use_random_ptr:
        rand_bf16 = tl.load(
            random_or_seed_ptr + offsets,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.bfloat16)
    else:
        seed = tl.load(random_or_seed_ptr + seed_index)
        rand_bf16 = tl.rand(seed, offsets.to(tl.uint32)).to(tl.bfloat16)

    dropout_p_bf16 = tl.full((ROW_BLOCK, BLOCK_H), dropout_p, tl.float32).to(
        tl.bfloat16
    )
    keep = rand_bf16 > dropout_p_bf16
    tl.store(gt_ptr + offsets, keep, mask=mask)

    dropped = tl.where(keep, projected, 0.0).to(tl.bfloat16)
    scaled = _f32_mul(dropped.to(tl.float32), dropout_scale).to(tl.bfloat16)
    residual = tl.load(
        residual_ptr + offsets,
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)
    x = residual + scaled.to(tl.float32)
    tl.store(add_ptr + offsets, x, mask=mask)

    x_for_reduce = tl.where(mask, x, 0.0)
    mean = tl.sum(x_for_reduce, axis=1) / hidden
    centered = x - mean[:, None]
    variance = tl.sum(tl.where(mask, centered * centered, 0.0), axis=1) / hidden
    invstd = tl.rsqrt(variance + eps)

    normalized = centered * invstd[:, None]
    tl.store(norm_ptr + offsets, normalized, mask=mask)

    weight = tl.load(
        weight_ptr + cols,
        mask=col_mask,
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float32)
    bias = tl.load(
        bias_ptr + cols,
        mask=col_mask,
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float32)
    affine = normalized * weight[None, :] + bias[None, :]
    tl.store(out_ptr + offsets, affine.to(tl.bfloat16), mask=mask)
    tl.store(invstd_div_ptr + row_ids, invstd / hidden, mask=row_mask)


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


def _torch_rand_uniform_offset_advance(numel, *, device):
    props = torch.cuda.get_device_properties(device)
    block_size = 256
    blocks_per_sm = props.max_threads_per_multi_processor // block_size
    grid = min(
        props.multi_processor_count * blocks_per_sm,
        (int(numel) + block_size - 1) // block_size,
    )
    unroll = 4
    counter_offset = ((int(numel) - 1) // (block_size * grid * unroll) + 1) * 4
    return counter_offset * 2


def _inductor_random_for_eager_check(shape, seed, *, device):
    if torch.cuda.is_current_stream_capturing():
        return torch.ops.prims.inductor_random.default(shape, seed, "rand")

    numel = 1
    for dim in shape:
        numel *= int(dim)
    advance = _torch_rand_uniform_offset_advance(numel, device=device)
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
    projected,
    random_or_seed,
    residual,
    weight,
    bias,
    gt,
    add,
    normalized,
    out,
    invstd_div,
    *,
    use_random_ptr,
):
    grid = lambda meta: (triton.cdiv(N_ROWS, meta["ROW_BLOCK"]),)
    _dropout_residual_layernorm_kernel[grid](
        projected,
        random_or_seed,
        residual,
        weight,
        bias,
        gt,
        add,
        normalized,
        out,
        invstd_div,
        total_rows=N_ROWS,
        hidden=HIDDEN,
        seed_index=SEED_INDEX,
        dropout_p=DROPOUT_P,
        dropout_scale=DROPOUT_SCALE,
        eps=EPS,
        use_random_ptr=use_random_ptr,
        BLOCK_H=BLOCK_H,
    )


# cfc55f11: (T([8192,1024], bf16), T([49], i64), T([16,512,1024], f32), ...)
@oracle_impl(hardware="B200", point="cfc55f11")
def oracle_forward(inputs):
    global _SEEDED_VARIANT_READY

    (
        projected,
        seeds,
        residual,
        weight,
        bias,
        view_shape_arg,
        random_shape_arg,
        out_shape_arg,
    ) = inputs
    view_shape = _as_shape(view_shape_arg)
    random_shape = _as_shape(random_shape_arg)
    out_shape = _as_shape(out_shape_arg)

    gt = torch.empty_strided(
        view_shape,
        _contiguous_stride(view_shape),
        device=projected.device,
        dtype=torch.bool,
    )
    add = torch.empty_strided(
        view_shape,
        _contiguous_stride(view_shape),
        device=projected.device,
        dtype=torch.float32,
    )
    normalized = torch.empty_strided(
        view_shape,
        _contiguous_stride(view_shape),
        device=projected.device,
        dtype=torch.float32,
    )
    out = torch.empty_strided(
        out_shape,
        _contiguous_stride(out_shape),
        device=projected.device,
        dtype=torch.bfloat16,
    )
    invstd_div = torch.empty_strided(
        (BATCH, SEQ_LEN, 1),
        (SEQ_LEN, 1, 1),
        device=projected.device,
        dtype=torch.float32,
    )

    if torch.cuda.is_current_stream_capturing():
        _launch(
            projected,
            seeds,
            residual,
            weight,
            bias,
            gt,
            add,
            normalized,
            out,
            invstd_div,
            use_random_ptr=False,
        )
    else:
        if not _SEEDED_VARIANT_READY:
            _launch(
                projected,
                seeds,
                residual,
                weight,
                bias,
                gt,
                add,
                normalized,
                out,
                invstd_div,
                use_random_ptr=False,
            )
            _SEEDED_VARIANT_READY = True

        seed = torch.ops.prims.inductor_lookup_seed.default(seeds, SEED_INDEX)
        random = _inductor_random_for_eager_check(
            random_shape,
            seed,
            device=projected.device,
        )
        _launch(
            projected,
            random,
            residual,
            weight,
            bias,
            gt,
            add,
            normalized,
            out,
            invstd_div,
            use_random_ptr=True,
        )

    return gt, add, normalized, out, invstd_div
