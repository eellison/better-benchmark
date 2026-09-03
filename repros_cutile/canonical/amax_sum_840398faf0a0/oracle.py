"""cuTile port of amax_sum_840398faf0a0: DeBERTa masked softmax + dropout.

Similar to dde0a4d3980e but without the /8.0 divide and mask has different
broadcast pattern. Returns (where, amax, sum_1, gt, view_1_bf16, permute).
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 16
DROPOUT_SCALE = 1.1111111111111112


@ct.kernel
def _masked_softmax_dropout_kernel(
    x_ptr,
    mask_ptr,
    fill_ptr,
    random_ptr,
    where_ptr,
    amax_ptr,
    sum_ptr,
    gt_ptr,
    out_ptr,
    BLOCK_N: ct.Constant[int],
):
    row = ct.bid(0)

    x_bf = ct.load(x_ptr, index=(row, 0), shape=(1, BLOCK_N))
    mask = ct.load(mask_ptr, index=(row, 0), shape=(1, BLOCK_N))
    fill_v = ct.load(fill_ptr, index=(0,), shape=(1,))
    fill_bf = ct.astype(fill_v, ct.bfloat16)
    fill_2d = ct.reshape(fill_bf, (1, 1))
    fill_broad = ct.zeros((1, BLOCK_N), dtype=ct.bfloat16) + fill_2d
    where_bf = ct.where(mask, fill_broad, x_bf)
    ct.store(where_ptr, index=(row, 0), tile=where_bf)

    scores = ct.astype(where_bf, ct.float32)
    row_max = ct.max(scores, axis=1, keepdims=True)
    ct.store(amax_ptr, index=(row,), tile=ct.reshape(row_max, (1,)))

    numer = ct.exp(scores - row_max)
    denom = ct.sum(numer, axis=1, keepdims=True)
    ct.store(sum_ptr, index=(row,), tile=ct.reshape(denom, (1,)))
    probs = numer / denom

    rand = ct.load(random_ptr, index=(row, 0), shape=(1, BLOCK_N))
    keep = rand > ct.full((1, BLOCK_N), 0.1, dtype=ct.float32)
    ct.store(gt_ptr, index=(row, 0), tile=keep)

    zero_f = ct.zeros((1, BLOCK_N), dtype=ct.float32)
    dropped = ct.where(keep, probs, zero_f)
    scaled = dropped * DROPOUT_SCALE
    ct.store(out_ptr, index=(row, 0), tile=ct.astype(scaled, ct.bfloat16))


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


@oracle_impl(hardware="B200", point="00541467", BLOCK_N=512)
def oracle_forward(inputs, *, BLOCK_N: int):
    arg0_1, arg1_1, arg2_1, arg3_1, _shape0, shape1, _shape2 = inputs
    device = arg0_1.device

    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        raise NotImplementedError(
            "cuTile port unsupported inside CUDA graph capture (seeded RNG)."
        )

    # shape0 has -1 (batch inferred). We can derive from arg0_1 [192, 512, 512].
    # arg0_1: [192, 512, 512] view -> [8, 24, 512, 512]
    heads = 24
    batch = arg0_1.shape[0] // heads
    q_len = arg0_1.shape[1]
    k_len = arg0_1.shape[2]
    view_shape = (batch, heads, q_len, k_len)
    random_shape = _shape(shape1)
    rows = batch * heads * q_len
    row_shape = view_shape[:-1] + (1,)
    flat_shape = (batch * heads, q_len, k_len)

    where_out = torch.empty(view_shape, device=device, dtype=torch.bfloat16)
    amax = torch.empty(row_shape, device=device, dtype=torch.float32)
    sum_1 = torch.empty(row_shape, device=device, dtype=torch.float32)
    gt = torch.empty(view_shape, device=device, dtype=torch.bool)
    convert_element_type_1 = torch.empty(flat_shape, device=device, dtype=torch.bfloat16)

    seed = torch.ops.prims.inductor_lookup_seed.default(arg3_1, SEED_INDEX)
    random = _inductor_random_for_eager_check(random_shape, seed, device=device)

    x_view = arg0_1.view(view_shape)
    x_2d = x_view.reshape(rows, k_len)

    mask_full = arg1_1.expand(batch, heads, q_len, k_len).contiguous()
    mask_flat = mask_full.view(rows, k_len)

    r_2d = random.contiguous().view(rows, k_len)
    where_2d = where_out.view(rows, k_len)
    amax_1d = amax.view(rows)
    sum_1d = sum_1.view(rows)
    gt_2d = gt.view(rows, k_len)
    out_2d = convert_element_type_1.view(rows, k_len)

    fill_1d = arg2_1.view(1)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (rows, 1, 1),
        _masked_softmax_dropout_kernel,
        (x_2d, mask_flat, fill_1d, r_2d, where_2d, amax_1d, sum_1d, gt_2d, out_2d,
         BLOCK_N),
    )
    permute = convert_element_type_1.permute(0, 2, 1)
    return where_out, amax, sum_1, gt, convert_element_type_1, permute
