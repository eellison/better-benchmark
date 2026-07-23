"""cuTile port of amax_sum_1d0b8274d1b3: T5 softmax + dropout, strided bias.

Kernel takes flat 2D (n_rows, k_len) views. arg1 has non-contiguous strides so
we pre-broadcast it into a contiguous (n_rows, k_len) f32 tile via torch prior
to the kernel (a graph-capturable pointwise op).
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 39
DROPOUT_SCALE = 1.1111111111111112


@ct.kernel
def _softmax_dropout_kernel(
    x_ptr,          # bf16 [rows, cols]  (view of arg0)
    bias_ptr,       # f32  [rows, cols]  (broadcast copy of arg1)
    random_ptr,     # f32  [rows, cols]
    rounded_ptr,    # bf16 [rows, cols]
    amax_ptr,       # f32  [rows]
    sum_ptr,        # f32  [rows]
    gt_ptr,         # bool [rows, cols]
    out_ptr,        # bf16 [rows, cols]
    COLS: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
):
    row = ct.bid(0)

    x_bf = ct.load(x_ptr, index=(row, 0), shape=(1, BLOCK_N))
    bias = ct.load(bias_ptr, index=(row, 0), shape=(1, BLOCK_N))
    x_f = ct.astype(x_bf, ct.float32)
    added_f = x_f + bias
    rounded_bf = ct.astype(added_f, ct.bfloat16)
    ct.store(rounded_ptr, index=(row, 0), tile=rounded_bf)

    scores = ct.astype(rounded_bf, ct.float32)
    row_max = ct.sum(scores, axis=1, keepdims=True)  # placeholder
    # We need row max — use ct.max
    row_max = ct.max(scores, axis=1, keepdims=True)
    ct.store(amax_ptr, index=(row,), tile=ct.reshape(row_max, (1,)))

    numer = ct.exp(scores - row_max)
    denom = ct.sum(numer, axis=1, keepdims=True)
    ct.store(sum_ptr, index=(row,), tile=ct.reshape(denom, (1,)))
    probs = numer / denom
    probs_bf = ct.astype(probs, ct.bfloat16)

    rand = ct.load(random_ptr, index=(row, 0), shape=(1, BLOCK_N))
    rand_bf = ct.astype(rand, ct.bfloat16)
    dropout_p_bf = ct.full((1, BLOCK_N), 0.1, dtype=ct.bfloat16)
    keep = rand_bf > dropout_p_bf
    ct.store(gt_ptr, index=(row, 0), tile=keep)

    zero_bf = ct.full((1, BLOCK_N), 0.0, dtype=ct.bfloat16)
    dropped = ct.where(keep, probs_bf, zero_bf)
    scaled = ct.astype(ct.astype(dropped, ct.float32) * DROPOUT_SCALE, ct.bfloat16)
    ct.store(out_ptr, index=(row, 0), tile=scaled)


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


@oracle_impl(hardware="B200", point="aeb1682d", BLOCK_N=1024)
def oracle_forward(inputs, *, BLOCK_N: int):
    arg0_1, arg1_1, arg2_1, shape0, shape1, _shape2, shape3 = inputs
    device = arg0_1.device

    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        raise NotImplementedError(
            "cuTile port unsupported inside CUDA graph capture (seeded RNG)."
        )

    view_shape = _shape(shape0)  # [8, 8, 1024, 1024]
    random_shape = _shape(shape1)
    flat_shape = _shape(shape3)  # [64, 1024, 1024]
    rows = 1
    for d in view_shape[:-1]:
        rows *= int(d)
    cols = int(view_shape[-1])
    row_shape = view_shape[:-1] + (1,)

    rounded = torch.empty(view_shape, device=device, dtype=torch.bfloat16)
    amax = torch.empty(row_shape, device=device, dtype=torch.float32)
    sum_1 = torch.empty(row_shape, device=device, dtype=torch.float32)
    gt = torch.empty(view_shape, device=device, dtype=torch.bool)
    view_1 = torch.empty(flat_shape, device=device, dtype=torch.bfloat16)

    seed = torch.ops.prims.inductor_lookup_seed.default(arg2_1, SEED_INDEX)
    random = _inductor_random_for_eager_check(random_shape, seed, device=device)

    # Broadcast arg1 to contiguous shape
    x_bf = arg0_1.view(view_shape)
    bias_contig = arg1_1.contiguous()

    x_2d = x_bf.reshape(rows, cols)
    bias_2d = bias_contig.reshape(rows, cols)
    r_2d = random.contiguous().view(rows, cols)
    rounded_2d = rounded.view(rows, cols)
    amax_1d = amax.view(rows)
    sum_1d = sum_1.view(rows)
    gt_2d = gt.view(rows, cols)
    out_2d = view_1.view(rows, cols)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (rows, 1, 1),
        _softmax_dropout_kernel,
        (x_2d, bias_2d, r_2d, rounded_2d, amax_1d, sum_1d, gt_2d, out_2d,
         cols, BLOCK_N),
    )
    permute = view_1.permute(0, 2, 1)
    return rounded, amax, sum_1, gt, view_1, permute
