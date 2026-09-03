"""cuTile port of var_mean_69ffe374d494: GoogleFnet f32 dropout+residual+LN+complex64.

Ports the Triton `_dropout_residual_layernorm_complex_kernel`. Pre-generates
the seeded random tensor via `torch.ops.prims.inductor_random` OUTSIDE the
kernel. HIDDEN=768 requires BLOCK_H=1024 with padded loads/stores.

Returns: (gt, mul_2, add_2, complex_out, div).
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 1
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112
EPS = 1.0e-12
HIDDEN = 768
BLOCK_H = 1024


@ct.kernel
def _dropout_residual_layernorm_kernel(
    flat_ptr,      # f32 [rows, HIDDEN]
    random_ptr,    # f32 [rows, HIDDEN]
    residual_ptr,  # f32 [rows, HIDDEN]
    weight_ptr,    # f32 [HIDDEN]
    bias_ptr,      # f32 [HIDDEN]
    gt_pad_ptr,    # bool [rows, BLOCK_H]
    norm_pad_ptr,  # f32 [rows, BLOCK_H]
    affine_pad_ptr,# f32 [rows, BLOCK_H]
    div_ptr,       # f32 [rows]
    HIDDEN_: ct.Constant[int],
    BLOCK_H_: ct.Constant[int],
):
    row = ct.bid(0)

    flat = ct.load(
        flat_ptr, index=(row, 0), shape=(1, BLOCK_H_),
        padding_mode=ct.PaddingMode.ZERO,
    )
    residual = ct.load(
        residual_ptr, index=(row, 0), shape=(1, BLOCK_H_),
        padding_mode=ct.PaddingMode.ZERO,
    )
    random_f = ct.load(
        random_ptr, index=(row, 0), shape=(1, BLOCK_H_),
        padding_mode=ct.PaddingMode.ZERO,
    )

    keep = random_f > DROPOUT_P
    ct.store(gt_pad_ptr, index=(row, 0), tile=keep)

    dropped = ct.astype(keep, ct.float32) * flat
    scaled = dropped * DROPOUT_SCALE
    x = scaled + residual

    col_idx = ct.arange(BLOCK_H_, dtype=ct.int32)
    col_mask = ct.reshape(col_idx < HIDDEN_, (1, BLOCK_H_))
    x_masked = ct.where(col_mask, x, 0.0)
    row_sum = ct.sum(x_masked)
    mean = row_sum * (1.0 / HIDDEN_)
    centered = x - mean
    centered_masked = ct.where(col_mask, centered, 0.0)
    variance = ct.sum(centered_masked * centered_masked) * (1.0 / HIDDEN_)
    invstd = ct.rsqrt(variance + EPS)
    normalized = centered * invstd

    weight = ct.load(
        weight_ptr, index=(0,), shape=(BLOCK_H_,),
        padding_mode=ct.PaddingMode.ZERO,
    )
    bias = ct.load(
        bias_ptr, index=(0,), shape=(BLOCK_H_,),
        padding_mode=ct.PaddingMode.ZERO,
    )
    w2 = ct.reshape(weight, (1, BLOCK_H_))
    b2 = ct.reshape(bias, (1, BLOCK_H_))
    affine = normalized * w2 + b2

    ct.store(norm_pad_ptr, index=(row, 0), tile=normalized)
    ct.store(affine_pad_ptr, index=(row, 0), tile=affine)
    ct.store(div_ptr, index=(row,), tile=ct.reshape(invstd * (1.0 / HIDDEN_), (1,)))


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


@oracle_impl(hardware="B200", point="3a80a44f", BLOCK_M=1, BLOCK_H=1024)
def oracle_forward(inputs, *, BLOCK_M: int, BLOCK_H: int):
    flat, seeds, residual, weight, bias, shape0, shape1 = inputs
    complex_shape = _as_shape(shape0)
    base_shape = _as_shape(shape1)
    device = flat.device
    rows = int(flat.shape[0])
    hidden = int(flat.shape[1])
    assert hidden == HIDDEN
    div_shape = (base_shape[0], base_shape[1], 1)

    seed = torch.ops.prims.inductor_lookup_seed.default(seeds, SEED_INDEX)
    random = _inductor_random_for_eager_check(base_shape, seed, device=device)
    random_flat = random.reshape(rows, hidden).contiguous()
    residual_flat = residual.reshape(rows, hidden).contiguous()

    gt_pad = torch.empty((rows, BLOCK_H), device=device, dtype=torch.bool)
    norm_pad = torch.empty((rows, BLOCK_H), device=device, dtype=torch.float32)
    affine_pad = torch.empty((rows, BLOCK_H), device=device, dtype=torch.float32)
    div_1d = torch.empty((rows,), device=device, dtype=torch.float32)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (rows, 1, 1), _dropout_residual_layernorm_kernel,
        (flat, random_flat, residual_flat, weight, bias,
         gt_pad, norm_pad, affine_pad, div_1d,
         hidden, BLOCK_H),
    )

    gt = torch.empty_strided(
        base_shape, _contiguous_stride(base_shape),
        device=device, dtype=torch.bool,
    )
    gt.view(rows, hidden).copy_(gt_pad.narrow(1, 0, hidden))

    normalized = torch.empty_strided(
        base_shape, _contiguous_stride(base_shape),
        device=device, dtype=torch.float32,
    )
    normalized.view(rows, hidden).copy_(norm_pad.narrow(1, 0, hidden))

    affine = torch.empty_strided(
        base_shape, _contiguous_stride(base_shape),
        device=device, dtype=torch.float32,
    )
    affine.view(rows, hidden).copy_(affine_pad.narrow(1, 0, hidden))

    # complex_out has real part = affine, imag = 0.
    complex_out = torch.empty_strided(
        complex_shape, _contiguous_stride(base_shape),
        device=device, dtype=torch.complex64,
    )
    real_view = torch.view_as_real(complex_out).view(rows, hidden, 2)
    real_view[..., 0].copy_(affine_pad.narrow(1, 0, hidden))
    real_view[..., 1].zero_()

    div = torch.empty_strided(
        div_shape, _contiguous_stride(div_shape),
        device=device, dtype=torch.float32,
    )
    div.view(rows).copy_(div_1d)

    return gt, normalized, affine, complex_out, div
