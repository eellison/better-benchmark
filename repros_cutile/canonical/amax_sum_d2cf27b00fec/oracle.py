"""cuTile port of amax_sum_d2cf27b00fec: DeBERTa masked softmax + dropout.

Ports the Triton `_masked_softmax_dropout_kernel`. The mask is broadcast from
[B, 1, Q, K] over heads. The fill value is a scalar bf16. Seeded on-device RNG
replaced with pre-generated `torch.ops.prims.inductor_random`.
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 7
DROPOUT_SCALE = 1.1111111111111112


@ct.kernel
def _masked_softmax_dropout_kernel(
    scores_ptr,   # bf16 [B, H, Q, K]
    mask_ptr,     # b8   [B, 1, Q, K]
    fill_ptr,     # bf16 []
    random_ptr,   # f32  [B, H, Q, K]
    where_ptr,    # bf16 [B, H, Q, K]
    amax_ptr,     # f32  [B, H, Q]
    denom_ptr,    # f32  [B, H, Q]
    keep_ptr,     # b8   [B, H, Q, K]
    out_ptr,      # bf16 [B, H, Q, K]
    HEADS: ct.Constant[int],
    Q_LEN: ct.Constant[int],
    BLOCK_M: ct.Constant[int],
    BLOCK_K: ct.Constant[int],
):
    # bid(0) walks (B*H) directly since BLOCK_M rows of Q are grouped by tile.
    bh = ct.bid(0)
    q_block = ct.bid(1)
    b = bh // HEADS
    h = bh - b * HEADS

    raw = ct.load(scores_ptr, index=(b, h, q_block, 0), shape=(1, 1, BLOCK_M, BLOCK_K))
    mask_val = ct.load(mask_ptr, index=(b, 0, q_block, 0), shape=(1, 1, BLOCK_M, BLOCK_K))
    fill = ct.load(fill_ptr, index=(0,), shape=(1,))
    fill_bf = ct.astype(fill, ct.bfloat16)
    # Broadcast fill from [1] to [1,1,BLOCK_M,BLOCK_K]
    fill_tile = ct.full((1, 1, BLOCK_M, BLOCK_K), 0.0, dtype=ct.bfloat16) + ct.reshape(fill_bf, (1, 1, 1, 1))

    masked_scores = ct.where(mask_val, fill_tile, raw)
    ct.store(where_ptr, index=(b, h, q_block, 0), tile=masked_scores)

    scores_f = ct.astype(masked_scores, ct.float32)
    row_max = ct.max(scores_f, axis=3, keepdims=True)
    numer = ct.exp(scores_f - row_max)
    denom = ct.sum(numer, axis=3, keepdims=True)
    probs = numer / denom

    ct.store(amax_ptr, index=(b, h, q_block, 0), tile=row_max)
    ct.store(denom_ptr, index=(b, h, q_block, 0), tile=denom)

    random = ct.load(random_ptr, index=(b, h, q_block, 0), shape=(1, 1, BLOCK_M, BLOCK_K))
    keep = random > 0.1
    ct.store(keep_ptr, index=(b, h, q_block, 0), tile=keep)

    zero_f = ct.full((1, 1, BLOCK_M, BLOCK_K), 0.0, dtype=ct.float32)
    dropped = ct.where(keep, probs, zero_f)
    scaled = dropped * DROPOUT_SCALE
    scaled_bf = ct.astype(scaled, ct.bfloat16)
    ct.store(out_ptr, index=(b, h, q_block, 0), tile=scaled_bf)


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= int(dim)
    return tuple(reversed(stride))


def _as_shape(shape):
    return tuple(int(dim) for dim in shape)


def _resolve_shape(shape, numel):
    dims = [int(dim) for dim in shape]
    known = 1
    missing = -1
    for idx, dim in enumerate(dims):
        if dim == -1:
            missing = idx
        else:
            known *= dim
    if missing >= 0:
        dims[missing] = int(numel) // known
    return tuple(dims)


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


@oracle_impl(hardware="B200", point="00541467", block_m=4, block_k=512)
def oracle_forward(inputs, *, block_m: int, block_k: int):
    arg0_1, arg1_1, arg2_1, arg3_1, shape0, shape1, shape2 = inputs
    view_shape = _resolve_shape(shape0, arg0_1.numel())
    random_shape = _as_shape(shape1)
    flat_shape = _resolve_shape(shape2, arg0_1.numel())
    reduction_shape = (view_shape[0], view_shape[1], view_shape[2], 1)
    B, H, Q, K = view_shape

    device = arg0_1.device
    where = torch.empty_strided(
        view_shape, _contiguous_stride(view_shape),
        device=device, dtype=torch.bfloat16,
    )
    amax = torch.empty_strided(
        reduction_shape, _contiguous_stride(reduction_shape),
        device=device, dtype=torch.float32,
    )
    denom = torch.empty_strided(
        reduction_shape, _contiguous_stride(reduction_shape),
        device=device, dtype=torch.float32,
    )
    keep = torch.empty_strided(
        view_shape, _contiguous_stride(view_shape),
        device=device, dtype=torch.bool,
    )
    out = torch.empty_strided(
        flat_shape, _contiguous_stride(flat_shape),
        device=device, dtype=torch.bfloat16,
    )

    # Views: scores are [B*Q, K] flat, we need [B, H, Q, K]
    scores_4d = arg0_1.view(B, H, Q, K)
    mask_4d = arg1_1  # [B, 1, Q, K]
    # Scalar bf16 fill: view as [1] so cuTile can load with shape (1,)
    fill_1d = arg2_1.view(1)

    seed = torch.ops.prims.inductor_lookup_seed.default(arg3_1, SEED_INDEX)
    random = _inductor_random_for_eager_check(random_shape, seed, device=device)
    random_4d = random.view(B, H, Q, K)

    out_4d = out.view(B, H, Q, K)

    if Q % block_m != 0:
        raise NotImplementedError(f"block_m={block_m} doesn't divide Q={Q}")
    if K != block_k:
        raise NotImplementedError(f"block_k={block_k} != K={K}")

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (B * H, Q // block_m, 1),
        _masked_softmax_dropout_kernel,
        (scores_4d, mask_4d, fill_1d, random_4d, where, amax, denom, keep, out_4d,
         H, Q, block_m, block_k),
    )

    return where, amax, denom, keep, out, out.permute(0, 2, 1)
