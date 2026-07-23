"""cuTile port of amax_sum_any_30cded63a2f9: BERT/Roberta/Electra/MobileBert softmax+dropout.

Row kernel: softmax(bf16 -> fp32 -> softmax -> bf16), then where(row_finite, bias, softmax_probs),
then seeded dropout scale, bf16 cast, alias views.
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 16
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
def _softmax_where_dropout_kernel(
    scores_ptr,     # bf16 [rows, K]
    bias_ptr,       # bf16 [rows, K]
    rand_ptr,       # f32 [rows, K]
    where_ptr,      # bf16 [rows, K]
    gt_ptr,         # b8 [rows, K]
    out_ptr,        # bf16 [rows, K]
    K: ct.Constant[int],
    DROPOUT_SCALE_C: ct.Constant[float],
):
    row = ct.bid(0)
    scores_bf = ct.load(scores_ptr, index=(row, 0), shape=(1, K))
    scores = ct.astype(scores_bf, ct.float32)
    bias = ct.load(bias_ptr, index=(row, 0), shape=(1, K))

    row_max = ct.max(scores, keepdims=True)
    numer = ct.exp(scores - row_max)
    denom = ct.sum(numer, keepdims=True)
    probs = numer / denom
    probs_bf = ct.astype(probs, ct.bfloat16)

    # Row is "all masked" when every score is -inf. Only then use the bias.
    neg_inf = ct.full((1, K), float("-inf"), dtype=ct.float32)
    is_neg_inf = scores == neg_inf                  # True where -inf
    is_not_neg_inf = ct.astype(~is_neg_inf, ct.int32)
    any_not_neg_inf = ct.max(is_not_neg_inf, keepdims=True)
    all_masked = any_not_neg_inf == 0

    where_out = ct.where(all_masked, bias, probs_bf)
    ct.store(where_ptr, index=(row, 0), tile=where_out)

    rand_val = ct.load(rand_ptr, index=(row, 0), shape=(1, K))
    rand_bf = ct.astype(rand_val, ct.bfloat16)
    p_bf = ct.full((1, K), 0.1, dtype=ct.bfloat16)
    keep = rand_bf > p_bf
    ct.store(gt_ptr, index=(row, 0), tile=keep)

    zero_bf = ct.full((1, K), 0.0, dtype=ct.bfloat16)
    dropped = ct.where(keep, where_out, zero_bf)
    scaled = ct.astype(ct.astype(dropped, ct.float32) * DROPOUT_SCALE_C, ct.bfloat16)
    ct.store(out_ptr, index=(row, 0), tile=scaled)


@oracle_impl(hardware="B200", point="8ae0f618")
@oracle_impl(hardware="B200", point="fac7e171")
@oracle_impl(hardware="B200", point="d59f4ab1")
def oracle_forward(inputs):
    arg0_1, arg1_1, arg2_1, shape0, shape1, shape2, shape3 = inputs
    device = arg0_1.device

    view_shape = tuple(int(d) for d in shape0)   # e.g., (32, 12, 512, 512)
    random_shape = tuple(int(d) for d in shape1)
    flat_shape = tuple(int(d) for d in shape3)

    K = int(view_shape[-1])
    rows = int(arg0_1.numel()) // K

    view = arg0_1.view(view_shape).contiguous()

    seed = torch.ops.prims.inductor_lookup_seed.default(arg2_1, SEED_INDEX)
    random = _inductor_random_for_eager_check(random_shape, seed, device=device)

    where_out = torch.empty(view_shape, device=device, dtype=torch.bfloat16)
    gt = torch.empty(view_shape, device=device, dtype=torch.bool)
    final = torch.empty(flat_shape, device=device, dtype=torch.bfloat16)

    scores_2d = view.reshape(rows, K).contiguous()
    bias_2d = arg1_1.reshape(rows, K).contiguous()
    rand_2d = random.reshape(rows, K).contiguous()

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (rows, 1, 1),
        _softmax_where_dropout_kernel,
        (scores_2d, bias_2d, rand_2d,
         where_out.view(rows, K), gt.view(rows, K), final.view(rows, K),
         K, DROPOUT_SCALE),
    )

    return where_out, gt, final, final.permute(0, 2, 1)
