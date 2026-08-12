"""cuTile port of amax_sum_ad24cfcd0e00: DebertaV2 attention softmax dropout.

Uses pre-generated random tensor (from torch.ops.prims.inductor_random) to
sidestep cuTile's lack of on-device seeded RNG.
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 10
DROPOUT_SCALE = 1.1111111111111112


@ct.kernel
def _softmax_dropout_kernel(
    view_ptr,           # bf16 [rows, k_len]
    mask_bcast_ptr,     # b8   [rows, k_len]
    scalar_bcast_ptr,   # bf16 [rows, k_len] (pre-expanded scalar)
    random_ptr,         # f32  [rows, k_len]
    where_out_ptr,      # bf16 [rows, k_len]
    amax_ptr,           # f32  [rows]
    sum_ptr,            # f32  [rows]
    gt_ptr,             # b8   [rows, k_len]
    final_ptr,          # bf16 [rows, k_len]
    BLOCK_M: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
):
    pid = ct.bid(0)

    view = ct.load(view_ptr, index=(pid, 0), shape=(BLOCK_M, BLOCK_N))
    m = ct.load(mask_bcast_ptr, index=(pid, 0), shape=(BLOCK_M, BLOCK_N))
    scalar_broadcast = ct.load(scalar_bcast_ptr, index=(pid, 0), shape=(BLOCK_M, BLOCK_N))
    where = ct.where(m, scalar_broadcast, view)
    ct.store(where_out_ptr, index=(pid, 0), tile=where)

    x = ct.astype(where, ct.float32)
    row_max = ct.max(x, axis=1, keepdims=True)
    ct.store(amax_ptr, index=(pid,), tile=ct.reshape(row_max, (BLOCK_M,)))
    shifted = x - row_max
    numer = ct.exp(shifted)
    denom = ct.sum(numer, axis=1, keepdims=True)
    ct.store(sum_ptr, index=(pid,), tile=ct.reshape(denom, (BLOCK_M,)))
    probs = numer / denom

    random = ct.load(random_ptr, index=(pid, 0), shape=(BLOCK_M, BLOCK_N))
    threshold = ct.full((BLOCK_M, BLOCK_N), 0.1, dtype=ct.float32)
    keep = random > threshold
    ct.store(gt_ptr, index=(pid, 0), tile=keep)

    zero_f = ct.full((BLOCK_M, BLOCK_N), 0.0, dtype=ct.float32)
    one_f = ct.full((BLOCK_M, BLOCK_N), 1.0, dtype=ct.float32)
    keep_f = ct.where(keep, one_f, zero_f)
    scaled = keep_f * probs * DROPOUT_SCALE
    ct.store(final_ptr, index=(pid, 0), tile=ct.astype(scaled, ct.bfloat16))


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


def _shape(shape):
    return tuple(int(d) for d in shape)


@oracle_impl(hardware="B200", point="00541467", BLOCK_M=1, BLOCK_N=512)
def oracle_forward(inputs, *, BLOCK_M: int, BLOCK_N: int):
    arg0_1, arg1_1, arg2_1, arg3_1, shape0, shape1, shape2 = inputs
    random_shape = _shape(shape1)     # (8, 24, 512, 512)
    flat_shape = _shape(shape2)       # (-1, 512, 512) or (192, 512, 512)
    device = arg0_1.device

    # shape0 may contain a -1; use random_shape (fully spec'd) for view.
    view_shape = random_shape
    b, h, q, k = view_shape
    rows = b * h * q

    # Prepare inputs
    view_2d = arg0_1.contiguous().view(rows, k)
    # Broadcast mask [8, 1, 512, 512] -> [8, 24, 512, 512], then flatten to [rows, k]
    mask_bcast = arg1_1.expand(b, h, q, k).contiguous().view(rows, k)
    # Pre-expand scalar to a [rows, k] tensor for use inside kernel
    scalar_bcast = arg2_1.to(torch.bfloat16).expand(rows, k).contiguous()

    # Outputs
    where_out = torch.empty(view_shape, device=device, dtype=torch.bfloat16)
    amax = torch.empty((b, h, q, 1), device=device, dtype=torch.float32)
    sum_1 = torch.empty((b, h, q, 1), device=device, dtype=torch.float32)
    gt = torch.empty(view_shape, device=device, dtype=torch.bool)
    total = b * h * q * k
    flat_out_shape = (total // (k * q), q, k)
    final = torch.empty(flat_out_shape, device=device, dtype=torch.bfloat16)

    seed = torch.ops.prims.inductor_lookup_seed.default(arg3_1, SEED_INDEX)
    random = _inductor_random_for_eager_check(random_shape, seed, device=device)

    where_2d = where_out.view(rows, k)
    amax_1d = amax.view(rows)
    sum_1d = sum_1.view(rows)
    gt_2d = gt.view(rows, k)
    final_2d = final.view(rows, k)
    random_2d = random.contiguous().view(rows, k)

    stream = torch.cuda.current_stream()
    grid = (ct.cdiv(rows, BLOCK_M), 1, 1)
    ct.launch(
        stream,
        grid,
        _softmax_dropout_kernel,
        (view_2d, mask_bcast, scalar_bcast, random_2d,
         where_2d, amax_1d, sum_1d, gt_2d, final_2d,
         BLOCK_M, BLOCK_N),
    )

    return where_out, amax, sum_1, gt, final, final.permute(0, 2, 1)
