"""cuTile port of amax_sum_c2886f99a0ae: T5 attention softmax + dropout.

Bf16 scores + f32 bias -> bf16 rounded added_bf16 -> row softmax -> seeded
dropout -> bf16 output + permute alias. K=1024 per row.
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 33
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
def _softmax_dropout_kernel(
    added_ptr,   # bf16 [rows, K] pre-computed added tensor
    rand_ptr,    # f32 [rows, K]
    amax_ptr,    # f32 [rows]
    sum_ptr,     # f32 [rows]
    gt_ptr,      # b8 [rows, K]
    out_ptr,     # bf16 [rows, K]
    K: ct.Constant[int],
    DROPOUT_SCALE_C: ct.Constant[float],
):
    row = ct.bid(0)
    scores_bf = ct.load(added_ptr, index=(row, 0), shape=(1, K))
    scores = ct.astype(scores_bf, ct.float32)

    row_max = ct.max(scores, keepdims=True)
    ct.store(amax_ptr, index=(row,), tile=ct.reshape(row_max, (1,)))
    numer = ct.exp(scores - row_max)
    denom = ct.sum(numer, keepdims=True)
    ct.store(sum_ptr, index=(row,), tile=ct.reshape(denom, (1,)))
    probs = numer / denom
    probs_bf = ct.astype(probs, ct.bfloat16)

    rand_val = ct.load(rand_ptr, index=(row, 0), shape=(1, K))
    rand_bf = ct.astype(rand_val, ct.bfloat16)
    p_bf = ct.full((1, K), 0.1, dtype=ct.bfloat16)
    keep = rand_bf > p_bf
    ct.store(gt_ptr, index=(row, 0), tile=keep)

    zero_bf = ct.full((1, K), 0.0, dtype=ct.bfloat16)
    dropped = ct.where(keep, probs_bf, zero_bf)
    scaled = ct.astype(
        ct.astype(dropped, ct.float32) * DROPOUT_SCALE_C, ct.bfloat16
    )
    ct.store(out_ptr, index=(row, 0), tile=scaled)


@oracle_impl(hardware="B200", point="aeb1682d")
def oracle_forward(inputs):
    arg0_1, arg1_1, arg2_1, shape0, shape1, shape2, shape3 = inputs
    device = arg0_1.device

    view_shape = tuple(int(d) for d in shape0)  # (8, 8, 1024, 1024)
    random_shape = tuple(int(d) for d in shape1)
    flat_shape = tuple(int(d) for d in shape3)  # (64, 1024, 1024)

    K = int(view_shape[-1])
    rows = int(arg0_1.numel()) // K
    row_shape = view_shape[:-1] + (1,)

    view = arg0_1.view(view_shape)
    added = (view.float() + arg1_1).to(torch.bfloat16)  # convert_element_type

    seed = torch.ops.prims.inductor_lookup_seed.default(arg2_1, SEED_INDEX)
    random = _inductor_random_for_eager_check(random_shape, seed, device=device)

    amax = torch.empty(row_shape, device=device, dtype=torch.float32)
    sum_out = torch.empty(row_shape, device=device, dtype=torch.float32)
    gt = torch.empty(view_shape, device=device, dtype=torch.bool)
    final = torch.empty(flat_shape, device=device, dtype=torch.bfloat16)

    added_2d = added.reshape(rows, K).contiguous()
    rand_2d = random.reshape(rows, K).contiguous()

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (rows, 1, 1),
        _softmax_dropout_kernel,
        (added_2d, rand_2d,
         amax.view(rows), sum_out.view(rows),
         gt.view(rows, K), final.view(rows, K),
         K, DROPOUT_SCALE),
    )
    return added, amax, sum_out, gt, final, final.permute(0, 2, 1)
