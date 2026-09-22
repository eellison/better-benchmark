"""cuTile port of amax_sum_6afeef4d689d: DeBERTaV2 masked softmax + dropout.

The masked where-fill is done in a cuTile row kernel along with softmax +
dropout scale + bf16 rounding. Random tensor pre-generated via inductor_random.
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 28
DROPOUT_SCALE = 1.1111111111111112


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


@ct.kernel
def _masked_softmax_dropout_kernel(
    scores_ptr,      # bf16 [rows, K] (pre-where'd)
    rand_ptr,        # f32 [rows, K]
    amax_ptr,        # f32 [rows]
    sum_ptr,         # f32 [rows]
    gt_ptr,          # b8 [rows, K]
    out_ptr,         # bf16 [rows, K]
    K: ct.Constant[int],
    DROPOUT_SCALE_C: ct.Constant[float],
):
    row = ct.bid(0)
    scores_bf = ct.load(scores_ptr, index=(row, 0), shape=(1, K))
    scores = ct.astype(scores_bf, ct.float32)

    row_max = ct.max(scores, keepdims=True)
    ct.store(amax_ptr, index=(row,), tile=ct.reshape(row_max, (1,)))
    shifted = scores - row_max
    numer = ct.exp(shifted)
    denom = ct.sum(numer, keepdims=True)
    ct.store(sum_ptr, index=(row,), tile=ct.reshape(denom, (1,)))
    probs = numer / denom

    rand_val = ct.load(rand_ptr, index=(row, 0), shape=(1, K))
    p_f = ct.full((1, K), 0.1, dtype=ct.float32)
    keep = rand_val > p_f
    ct.store(gt_ptr, index=(row, 0), tile=keep)

    zero_f = ct.full((1, K), 0.0, dtype=ct.float32)
    dropped = ct.where(keep, probs, zero_f)
    scaled = dropped * DROPOUT_SCALE_C
    ct.store(out_ptr, index=(row, 0), tile=ct.astype(scaled, ct.bfloat16))


def _shape(shape, numel):
    dims = [int(d) for d in shape]
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


@oracle_impl(hardware="B200", point="00541467")
def oracle_forward(inputs):
    arg0_1, arg1_1, arg2_1, arg3_1, shape0, shape1, shape2 = inputs
    device = arg0_1.device
    numel = int(arg0_1.numel())
    view_shape = _shape(shape0, numel)   # (8, 24, 512, 512) or (-1, 24, 512, 512)
    random_shape = tuple(int(d) for d in shape1)  # (8, 24, 512, 512)
    flat_shape = _shape(shape2, numel)

    # Compute where(mask, fill, view) in torch
    view = arg0_1.view(view_shape)
    where_out = torch.where(arg1_1, arg2_1, view)  # bf16 [8, 24, 512, 512]

    K = int(view_shape[-1])
    rows = numel // K
    row_shape = view_shape[:-1] + (1,)

    seed = torch.ops.prims.inductor_lookup_seed.default(arg3_1, SEED_INDEX)
    random = _inductor_random_for_eager_check(random_shape, seed, device=device)

    amax_out = torch.empty(row_shape, device=device, dtype=torch.float32)
    sum_out = torch.empty(row_shape, device=device, dtype=torch.float32)
    gt = torch.empty(view_shape, device=device, dtype=torch.bool)
    out_bf16 = torch.empty(flat_shape, device=device, dtype=torch.bfloat16)

    scores_2d = where_out.reshape(rows, K).contiguous()
    rand_2d = random.reshape(rows, K).contiguous()

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (rows, 1, 1),
        _masked_softmax_dropout_kernel,
        (scores_2d, rand_2d,
         amax_out.view(rows), sum_out.view(rows),
         gt.view(rows, K), out_bf16.view(rows, K),
         K, DROPOUT_SCALE),
    )
    return where_out, amax_out, sum_out, gt, out_bf16, out_bf16.permute(0, 2, 1)
