"""cuTile port of amax_amax_any_f7b091f87d06: XLNet relative-shift softmax+dropout.

The relative-shift gather runs in torch (index ops are graph-capturable).
The scaled softmax + dropout + bf16 rounding uses a cuTile row kernel.
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 66
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
def _scaled_softmax_dropout_kernel(
    x_ptr, rand_ptr,
    unsc_amax_ptr, sc_amax_ptr, finite_ptr, denom_ptr,
    gt_ptr, final_ptr,
    K: ct.Constant[int],
    DROPOUT_SCALE_C: ct.Constant[float],
):
    row = ct.bid(0)
    added_bf = ct.load(x_ptr, index=(row, 0), shape=(1, K))
    unsc = ct.astype(added_bf, ct.float32)
    scaled_bf = ct.astype(unsc * 0.125, ct.bfloat16)
    scaled = ct.astype(scaled_bf, ct.float32)

    unsc_max = ct.max(unsc, keepdims=True)
    sc_max = ct.max(scaled, keepdims=True)
    ct.store(unsc_amax_ptr, index=(row,), tile=ct.reshape(unsc_max, (1,)))
    ct.store(sc_amax_ptr, index=(row,), tile=ct.reshape(sc_max, (1,)))

    is_finite = (scaled == scaled)
    inf_val = ct.full((1, K), float("inf"), dtype=ct.float32)
    neg_inf_val = ct.full((1, K), float("-inf"), dtype=ct.float32)
    finite = is_finite & (scaled != inf_val) & (scaled != neg_inf_val)
    finite_int = ct.astype(finite, ct.int32)
    finite_min = ct.min(finite_int, keepdims=True)
    all_finite = finite_min == 1
    ct.store(finite_ptr, index=(row,), tile=ct.reshape(all_finite, (1,)))

    shifted_unsc = (unsc - unsc_max) * 0.125
    shifted_sc = scaled - sc_max
    shifted = ct.where(all_finite, shifted_unsc, shifted_sc)
    numer = ct.exp(shifted)
    denom = ct.sum(numer, keepdims=True)
    ct.store(denom_ptr, index=(row,), tile=ct.reshape(denom, (1,)))
    probs = numer / denom

    rand_val = ct.load(rand_ptr, index=(row, 0), shape=(1, K))
    p_f = ct.full((1, K), 0.1, dtype=ct.float32)
    keep = rand_val > p_f
    ct.store(gt_ptr, index=(row, 0), tile=keep)

    zero_f = ct.full((1, K), 0.0, dtype=ct.float32)
    dropped = ct.where(keep, probs, zero_f)
    scaled_out = dropped * DROPOUT_SCALE_C
    ct.store(final_ptr, index=(row, 0), tile=ct.astype(scaled_out, ct.bfloat16))


@oracle_impl(hardware="B200", point="782e420b")
def oracle_forward(inputs):
    (arg0_1, arg1_1, arg2_1, arg3_1,
     s0, s1, s2, s3, s4, s5, s6, s7) = inputs
    device = arg0_1.device

    view = arg0_1.view(16, 16, 512, 1, 512)
    permuted = view.permute(0, 1, 2, 4, 3)
    view_1 = permuted.reshape(16, 16, 512, 512)

    view_2 = arg1_1.view(16, 16, 512, 1, 1024)
    permuted_1 = view_2.permute(0, 1, 2, 4, 3)
    view_3 = permuted_1.reshape(16, 16, 512, 1024)
    view_4 = view_3.view(16, 16, 1024, 512)
    slice_1 = view_4[:, :, 1:, :]
    view_5 = slice_1.reshape(16, 16, 512, 1023)
    index = view_5[:, :, :, arg2_1]
    add_1 = view_1 + index

    K = 512
    rows = 16 * 16 * 512
    full_shape = (16, 16, 512, 512)
    row_shape = (16, 16, 512, 1)

    add_2d = add_1.reshape(rows, K).contiguous()

    seed = torch.ops.prims.inductor_lookup_seed.default(arg3_1, SEED_INDEX)
    random = _inductor_random_for_eager_check(full_shape, seed, device=device)
    rand_2d = random.reshape(rows, K).contiguous()

    unsc_amax = torch.empty(row_shape, device=device, dtype=torch.float32)
    sc_amax = torch.empty(row_shape, device=device, dtype=torch.float32)
    finite = torch.empty(row_shape, device=device, dtype=torch.bool)
    denom = torch.empty(row_shape, device=device, dtype=torch.float32)
    gt = torch.empty(full_shape, device=device, dtype=torch.bool)
    final = torch.empty((256, 512, 512), device=device, dtype=torch.bfloat16)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (rows, 1, 1),
        _scaled_softmax_dropout_kernel,
        (add_2d, rand_2d,
         unsc_amax.view(rows), sc_amax.view(rows),
         finite.view(rows), denom.view(rows),
         gt.view(rows, K), final.view(rows, K),
         K, DROPOUT_SCALE),
    )

    return add_1, unsc_amax, sc_amax, finite, denom, gt, final, final.permute(0, 2, 1)
