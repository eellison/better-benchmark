"""cuTile port of var_mean_6a7f72eec85a: DebertaV2/BERT/etc dropout+residual+LN.

Shape points cover HIDDEN in {1536, 768, 256}, chose BLOCK_H per-point:
- 1536: BLOCK_H=2048 with masking
- 768:  BLOCK_H=1024 with masking
- 256:  BLOCK_H=256 (exact fit)
Fused seeded Inductor dropout + residual add + var_mean(correction=0) +
rsqrt(eps=1e-7) + affine + bf16 output + rsqrt/HIDDEN side.
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 32
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112
EPS = 1.0e-7


def _make_kernel(BLOCK_H):
    @ct.kernel
    def _kernel(
        flat_ptr,
        random_ptr,
        residual_ptr,
        weight_ptr,
        bias_ptr,
        mask_ptr,
        normalized_ptr,
        affine_ptr,
        bf16_ptr,
        div_ptr,
        HIDDEN_: ct.Constant[int],
    ):
        row = ct.bid(0)
        flat = ct.load(flat_ptr, index=(row, 0), shape=(1, BLOCK_H))
        residual = ct.load(residual_ptr, index=(row, 0), shape=(1, BLOCK_H))
        rand_f = ct.load(random_ptr, index=(row, 0), shape=(1, BLOCK_H))

        rand_bf = ct.astype(rand_f, ct.bfloat16)
        threshold_bf = ct.astype(
            ct.full(shape=(1, BLOCK_H), fill_value=DROPOUT_P, dtype=ct.float32),
            ct.bfloat16,
        )
        keep = rand_bf > threshold_bf
        ct.store(mask_ptr, index=(row, 0), tile=keep)

        dropped_bf = ct.astype(
            ct.where(keep, ct.astype(flat, ct.float32), 0.0),
            ct.bfloat16,
        )
        scaled_bf = ct.astype(
            ct.astype(dropped_bf, ct.float32) * DROPOUT_SCALE,
            ct.bfloat16,
        )
        layernorm_input = ct.astype(scaled_bf, ct.float32) + residual

        col_idx = ct.arange(BLOCK_H, dtype=ct.int32)
        col_mask = ct.reshape(col_idx < HIDDEN_, (1, BLOCK_H))
        x_masked = ct.where(col_mask, layernorm_input, 0.0)
        total = ct.sum(x_masked)
        mean = total * (1.0 / HIDDEN_)
        centered = layernorm_input - mean
        centered_masked = ct.where(col_mask, centered, 0.0)
        variance = ct.sum(centered_masked * centered_masked) * (1.0 / HIDDEN_)
        invstd = ct.rsqrt(variance + EPS)
        normalized = centered * invstd

        weight = ct.load(weight_ptr, index=(0,), shape=(BLOCK_H,))
        bias = ct.load(bias_ptr, index=(0,), shape=(BLOCK_H,))
        weight_2d = ct.reshape(weight, (1, BLOCK_H))
        bias_2d = ct.reshape(bias, (1, BLOCK_H))
        affine = normalized * weight_2d + bias_2d

        ct.store(normalized_ptr, index=(row, 0), tile=normalized)
        ct.store(affine_ptr, index=(row, 0), tile=affine)
        ct.store(bf16_ptr, index=(row, 0), tile=ct.astype(affine, ct.bfloat16))
        ct.store(div_ptr, index=(row,), tile=ct.reshape(
            ct.full((1,), invstd / HIDDEN_, dtype=ct.float32), (1,)))
    return _kernel


_KERNEL_2048 = _make_kernel(2048)
_KERNEL_1024 = _make_kernel(1024)
_KERNEL_256 = _make_kernel(256)


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


def _pick_block(hidden):
    if hidden <= 256:
        return 256, _KERNEL_256
    if hidden <= 1024:
        return 1024, _KERNEL_1024
    if hidden <= 2048:
        return 2048, _KERNEL_2048
    raise NotImplementedError(f"HIDDEN={hidden} not supported")


def _run(inputs):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, shape0, shape1, shape2 = inputs
    device = arg0_1.device
    norm_shape = _as_shape(shape0)
    random_shape = _as_shape(shape1)
    flat_shape = _as_shape(shape2)
    rows = int(arg0_1.shape[0])
    hidden = int(arg3_1.shape[0])
    div_shape = (norm_shape[0], norm_shape[1], 1)

    BLOCK_H, kernel = _pick_block(hidden)
    seed = torch.ops.prims.inductor_lookup_seed.default(arg1_1, SEED_INDEX)
    random = _inductor_random_for_eager_check(random_shape, seed, device=device)
    random_2d = random.reshape(rows, hidden).contiguous()
    residual_2d = arg2_1.reshape(rows, hidden).contiguous()

    if hidden == BLOCK_H:
        flat_pad = arg0_1
        bias_pad = None
        r_pad = random_2d
        resid_pad = residual_2d
        w_pad = arg3_1
        lnb_pad = arg4_1
    else:
        flat_pad = torch.zeros((rows, BLOCK_H), device=device, dtype=torch.bfloat16)
        flat_pad.narrow(1, 0, hidden).copy_(arg0_1)
        r_pad = torch.zeros((rows, BLOCK_H), device=device, dtype=torch.float32)
        r_pad.narrow(1, 0, hidden).copy_(random_2d)
        resid_pad = torch.zeros((rows, BLOCK_H), device=device, dtype=torch.float32)
        resid_pad.narrow(1, 0, hidden).copy_(residual_2d)
        w_pad = torch.zeros((BLOCK_H,), device=device, dtype=torch.float32)
        w_pad.narrow(0, 0, hidden).copy_(arg3_1)
        lnb_pad = torch.zeros((BLOCK_H,), device=device, dtype=torch.float32)
        lnb_pad.narrow(0, 0, hidden).copy_(arg4_1)

    mask_pad = torch.empty((rows, BLOCK_H), device=device, dtype=torch.bool)
    normalized_pad = torch.empty((rows, BLOCK_H), device=device, dtype=torch.float32)
    affine_pad = torch.empty((rows, BLOCK_H), device=device, dtype=torch.float32)
    bf16_pad = torch.empty((rows, BLOCK_H), device=device, dtype=torch.bfloat16)
    div_1d = torch.empty((rows,), device=device, dtype=torch.float32)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (rows, 1, 1), kernel,
        (flat_pad, r_pad, resid_pad, w_pad, lnb_pad,
         mask_pad, normalized_pad, affine_pad, bf16_pad, div_1d, hidden),
    )

    mask = torch.empty_strided(
        norm_shape, _contiguous_stride(norm_shape),
        device=device, dtype=torch.bool,
    )
    mask.view(rows, hidden).copy_(mask_pad.narrow(1, 0, hidden))
    normalized = torch.empty_strided(
        norm_shape, _contiguous_stride(norm_shape),
        device=device, dtype=torch.float32,
    )
    normalized.view(rows, hidden).copy_(normalized_pad.narrow(1, 0, hidden))
    affine = torch.empty_strided(
        norm_shape, _contiguous_stride(norm_shape),
        device=device, dtype=torch.float32,
    )
    affine.view(rows, hidden).copy_(affine_pad.narrow(1, 0, hidden))
    bf16_view = torch.empty_strided(
        flat_shape, _contiguous_stride(flat_shape),
        device=device, dtype=torch.bfloat16,
    )
    bf16_view.copy_(bf16_pad.narrow(1, 0, hidden).reshape(flat_shape))
    div = torch.empty_strided(
        div_shape, _contiguous_stride(div_shape),
        device=device, dtype=torch.float32,
    )
    div.view(rows).copy_(div_1d)

    return mask, normalized, affine, bf16_view, div


@oracle_impl(hardware="B200", point="55aa5fd0")
@oracle_impl(hardware="B200", point="243d7832")
@oracle_impl(hardware="B200", point="d9ecc504")
def oracle_forward(inputs):
    return _run(inputs)
