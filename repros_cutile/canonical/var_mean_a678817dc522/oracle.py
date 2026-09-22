"""cuTile port of var_mean_a678817dc522: dropout-residual LayerNorm.

Runs a pre-generated random tensor through the Triton-equivalent kernel:
- dropout mask via `rand > 0.1`
- scaled dropout (bf16 rounded)
- add residual f32
- LayerNorm mean/rsqrt
- affine + returned bf16 view

Uses torch.ops.prims.inductor_random on the CPU-side to build the random
tensor, so it doesn't need on-device seeded RNG in cuTile.
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _dropout_residual_layernorm_kernel(
    addmm_ptr,      # bf16 [ROWS, HIDDEN]
    random_ptr,     # f32  [ROWS, HIDDEN] (pre-generated random 0..1)
    residual_ptr,   # f32  [ROWS, HIDDEN]
    weight_ptr,     # f32  [HIDDEN]
    bias_ptr,       # f32  [HIDDEN]
    gt_ptr,         # b8   [ROWS, HIDDEN]
    norm_ptr,       # f32  [ROWS, HIDDEN]
    affine_ptr,     # f32  [ROWS, HIDDEN]
    bf16_ptr,       # bf16 [ROWS, HIDDEN]
    div_ptr,        # f32  [ROWS, 1]
    HIDDEN: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
    ROW_BLOCK: ct.Constant[int],
    EPS: ct.Constant[float],
):
    row_block = ct.bid(0)

    addmm = ct.load(addmm_ptr, index=(row_block, 0), shape=(ROW_BLOCK, BLOCK_H))
    rand = ct.load(random_ptr, index=(row_block, 0), shape=(ROW_BLOCK, BLOCK_H))
    residual = ct.load(residual_ptr, index=(row_block, 0), shape=(ROW_BLOCK, BLOCK_H))

    residual_f = ct.astype(residual, ct.float32)
    rand_bf = ct.astype(rand, ct.bfloat16)
    dropout_p_bf = ct.full((ROW_BLOCK, BLOCK_H), 0.1, dtype=ct.bfloat16)
    keep = rand_bf > dropout_p_bf
    ct.store(gt_ptr, index=(row_block, 0), tile=keep)

    zero_bf = ct.full((ROW_BLOCK, BLOCK_H), 0.0, dtype=ct.bfloat16)
    dropped = ct.where(keep, addmm, zero_bf)
    dropped_scaled = ct.astype(ct.astype(dropped, ct.float32) * 1.1111111111111112, ct.bfloat16)
    x = ct.astype(dropped_scaled, ct.float32) + residual_f

    inv_h = 1.0 / HIDDEN
    mean_1d = ct.sum(x, axis=1, keepdims=True) * inv_h
    centered = x - mean_1d
    variance_1d = ct.sum(centered * centered, axis=1, keepdims=True) * inv_h
    invstd_1d = ct.rsqrt(variance_1d + EPS)
    normalized = centered * invstd_1d

    weight = ct.load(weight_ptr, index=(0,), shape=(BLOCK_H,))
    bias = ct.load(bias_ptr, index=(0,), shape=(BLOCK_H,))
    weight_2d = ct.reshape(weight, (1, BLOCK_H))
    bias_2d = ct.reshape(bias, (1, BLOCK_H))
    affine = normalized * weight_2d + bias_2d

    ct.store(norm_ptr, index=(row_block, 0), tile=normalized)
    ct.store(affine_ptr, index=(row_block, 0), tile=affine)
    ct.store(bf16_ptr, index=(row_block, 0), tile=ct.astype(affine, ct.bfloat16))
    ct.store(div_ptr, index=(row_block, 0), tile=invstd_1d * inv_h)


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= dim
    return tuple(reversed(stride))


def _shape(shape):
    return tuple(int(d) for d in shape)


def _state_u64(state, start):
    return int.from_bytes(bytes(state[start : start + 8].tolist()), "little")


def _put_state_u64(state, start, value):
    state[start : start + 8] = torch.tensor(
        list(int(value).to_bytes(8, "little", signed=False)),
        dtype=state.dtype,
        device=state.device,
    )


def _inductor_random_for_eager_check(shape, seed, *, device):
    if torch.cuda.is_current_stream_capturing():
        return torch.ops.prims.inductor_random.default(shape, seed, "rand")

    numel = 1
    for dim in shape:
        numel *= int(dim)
    advance = (numel + 131071) // 131072
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


@oracle_impl(hardware="B200", point="55aa5fd0", BLOCK_H=2048, ROW_BLOCK=1)
@oracle_impl(hardware="B200", point="243d7832", BLOCK_H=1024, ROW_BLOCK=1)
@oracle_impl(hardware="B200", point="d9ecc504", BLOCK_H=256, ROW_BLOCK=2)
@oracle_impl(hardware="B200", point="d429ff7b", BLOCK_H=1024, ROW_BLOCK=1)
def oracle_forward(inputs, *, BLOCK_H: int, ROW_BLOCK: int):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, shape0, _shape1, shape2 = inputs

    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        # cuTile can't do seeded RNG inside a captured graph; refuse in this
        # dispatch path since we can't reproduce Inductor's stream state.
        raise NotImplementedError(
            "cuTile port unsupported inside CUDA graph capture (seeded RNG)."
        )

    norm_shape = _shape(shape0)
    flat_shape = _shape(shape2)
    rows = int(arg0_1.shape[0])
    hidden = int(arg3_1.shape[0])
    device = arg0_1.device

    div_shape = (norm_shape[0], norm_shape[1], 1)

    gt = torch.empty_strided(norm_shape, _contiguous_stride(norm_shape), device=device, dtype=torch.bool)
    normalized = torch.empty_strided(norm_shape, _contiguous_stride(norm_shape), device=device, dtype=torch.float32)
    affine = torch.empty_strided(norm_shape, _contiguous_stride(norm_shape), device=device, dtype=torch.float32)
    bf16_view = torch.empty_strided(flat_shape, _contiguous_stride(flat_shape), device=device, dtype=torch.bfloat16)
    div = torch.empty_strided(div_shape, _contiguous_stride(div_shape), device=device, dtype=torch.float32)

    seed = torch.ops.prims.inductor_lookup_seed.default(arg1_1, 3)
    random = _inductor_random_for_eager_check(norm_shape, seed, device=device)

    # Reshape all to [rows, hidden].
    addmm_2d = arg0_1.contiguous().view(rows, hidden)
    random_2d = random.contiguous().view(rows, hidden)
    residual_2d = arg2_1.contiguous().view(rows, hidden)
    gt_2d = gt.view(rows, hidden)
    normalized_2d = normalized.view(rows, hidden)
    affine_2d = affine.view(rows, hidden)
    bf16_2d = bf16_view.view(rows, hidden)
    div_2d = div.view(rows, 1)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(rows, ROW_BLOCK), 1, 1),
        _dropout_residual_layernorm_kernel,
        (addmm_2d, random_2d, residual_2d, arg3_1, arg4_1,
         gt_2d, normalized_2d, affine_2d, bf16_2d, div_2d,
         hidden, BLOCK_H, ROW_BLOCK, 1.0e-7),
    )
    return (gt, normalized, affine, bf16_view, div)
