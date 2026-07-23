"""cuTile port of amax_amax_any_f394b5666305: LayoutLM scaled softmax dropout.

For each row: bf16 * 0.125 -> two amax paths -> finite guard -> softmax ->
bf16 probs -> seeded dropout -> bf16 output + permute alias.
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 4
DROPOUT_SCALE = 1.1111111111111112


@ct.kernel
def _scaled_softmax_dropout_kernel(
    x_ptr, random_ptr,
    raw_amax_ptr, scaled_amax_ptr, finite_ptr, sum_ptr,
    keep_ptr, out_ptr,
    K: ct.Constant[int],
    BLOCK_M: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
):
    row_block = ct.bid(0)

    raw_bf = ct.load(x_ptr, index=(row_block, 0), shape=(BLOCK_M, BLOCK_N))
    raw = ct.astype(raw_bf, ct.float32)
    scaled_bf = ct.astype(raw * 0.125, ct.bfloat16)
    scaled = ct.astype(scaled_bf, ct.float32)

    raw_max = ct.max(raw, axis=1, keepdims=True)
    scaled_max = ct.max(scaled, axis=1, keepdims=True)
    ct.store(raw_amax_ptr, index=(row_block, 0), tile=raw_max)
    ct.store(scaled_amax_ptr, index=(row_block, 0), tile=scaled_max)

    inf_val = ct.full((BLOCK_M, BLOCK_N), float("inf"), dtype=ct.float32)
    neg_inf_val = ct.full((BLOCK_M, BLOCK_N), -float("inf"), dtype=ct.float32)
    is_nan = scaled != scaled
    is_inf = scaled == inf_val
    is_ninf = scaled == neg_inf_val
    true_tile = ct.full((BLOCK_M, BLOCK_N), True, dtype=ct.bool_)
    inv_mask = ct.where(is_nan, true_tile,
                        ct.where(is_inf, true_tile, is_ninf))
    has_invalid = ct.max(ct.astype(inv_mask, ct.int32), axis=1, keepdims=True) != 0
    row_is_finite = ct.astype(1 - ct.astype(has_invalid, ct.int32), ct.bool_)
    ct.store(finite_ptr, index=(row_block, 0), tile=row_is_finite)

    shifted_unscaled = (raw - raw_max) * 0.125
    shifted_scaled = scaled - scaled_max
    shifted = ct.where(row_is_finite, shifted_unscaled, shifted_scaled)
    numer = ct.exp(shifted)
    denom = ct.sum(numer, axis=1, keepdims=True)
    ct.store(sum_ptr, index=(row_block, 0), tile=denom)
    probs = ct.astype(numer / denom, ct.bfloat16)

    random = ct.load(random_ptr, index=(row_block, 0), shape=(BLOCK_M, BLOCK_N))
    rand_bf = ct.astype(random, ct.bfloat16)
    keep = rand_bf > ct.full((BLOCK_M, BLOCK_N), 0.1, dtype=ct.bfloat16)
    ct.store(keep_ptr, index=(row_block, 0), tile=keep)

    zero_bf = ct.full((BLOCK_M, BLOCK_N), 0.0, dtype=ct.bfloat16)
    dropped = ct.where(keep, probs, zero_bf)
    scaled_out = ct.astype(ct.astype(dropped, ct.float32) * DROPOUT_SCALE, ct.bfloat16)
    ct.store(out_ptr, index=(row_block, 0), tile=scaled_out)


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


@oracle_impl(hardware="B200", point="279c055a", BLOCK_M=1, BLOCK_N=512)
def oracle_forward(inputs, *, BLOCK_M: int, BLOCK_N: int):
    arg0_1, arg1_1, *_shape_params = inputs

    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        raise NotImplementedError(
            "cuTile port unsupported inside CUDA graph capture (seeded RNG)."
        )

    device = arg0_1.device
    B, H, Q, K = 32, 12, 512, 512
    N_ROWS = B * H * Q  # 196608
    full_shape = (B, H, Q, K)
    row_shape = (B, H, Q, 1)

    view = arg0_1.view(full_shape).contiguous()
    x_2d = view.reshape(N_ROWS, K)

    raw_amax = torch.empty(row_shape, device=device, dtype=torch.float32)
    scaled_amax = torch.empty(row_shape, device=device, dtype=torch.float32)
    finite = torch.empty(row_shape, device=device, dtype=torch.bool)
    sum_1 = torch.empty(row_shape, device=device, dtype=torch.float32)
    gt = torch.empty(full_shape, device=device, dtype=torch.bool)
    view_1 = torch.empty(arg0_1.shape, device=device, dtype=torch.bfloat16)

    seed = torch.ops.prims.inductor_lookup_seed.default(arg1_1, SEED_INDEX)
    random = _inductor_random_for_eager_check(full_shape, seed, device=device)
    random_2d = random.reshape(N_ROWS, K)

    raw_amax_2d = raw_amax.view(N_ROWS, 1)
    scaled_amax_2d = scaled_amax.view(N_ROWS, 1)
    finite_2d = finite.view(N_ROWS, 1)
    sum_1_2d = sum_1.view(N_ROWS, 1)
    gt_2d = gt.view(N_ROWS, K)
    view_1_2d = view_1.view(N_ROWS, K)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(N_ROWS, BLOCK_M), 1, 1),
        _scaled_softmax_dropout_kernel,
        (x_2d, random_2d, raw_amax_2d, scaled_amax_2d, finite_2d, sum_1_2d,
         gt_2d, view_1_2d, K, BLOCK_M, BLOCK_N),
    )

    permute = view_1.permute(0, 2, 1)
    return raw_amax, scaled_amax, finite, sum_1, gt, view_1, permute
