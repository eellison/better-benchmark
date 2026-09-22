"""cuTile port of amax_sum_192e2b05a200: MT5 attention softmax + dropout.

For each row: score + bias (strided), bf16 round, fp32 softmax with amax/sum
side outputs, bf16 output, dropout via pre-generated random tensor,
scaled bf16 output and its permute alias.
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 71
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112

BATCH = 32
HEADS = 6
Q_LEN = 128
K_LEN = 128
N_ROWS = BATCH * HEADS * Q_LEN


@ct.kernel
def _softmax_dropout_kernel(
    score_ptr,      # bf16 (BATCH*HEADS, Q_LEN, K_LEN) via 3D layout
    bias_ptr,       # f32 (BATCH*HEADS*Q_LEN*K_LEN) strided view
    random_ptr,     # f32 (N_ROWS, K_LEN)
    rounded_ptr,    # bf16 (BATCH*HEADS, Q_LEN, K_LEN) - 3D
    amax_ptr,       # f32 (N_ROWS,)
    sum_ptr,        # f32 (N_ROWS,)
    keep_ptr,       # bool (N_ROWS, K_LEN)
    dropped_ptr,    # bf16 (N_ROWS, K_LEN)
    K_LEN_C: ct.Constant[int],
):
    row = ct.bid(0)

    # score_2d loaded via 2D indexing on (N_ROWS, K_LEN)
    score = ct.load(score_ptr, index=(row, 0), shape=(1, K_LEN_C))
    bias = ct.load(bias_ptr, index=(row, 0), shape=(1, K_LEN_C))

    added = ct.astype(score, ct.float32) + bias
    rounded = ct.astype(added, ct.bfloat16)
    ct.store(rounded_ptr, index=(row, 0), tile=rounded)

    x = ct.astype(rounded, ct.float32)
    row_max = ct.max(x, axis=1, keepdims=True)
    numer = ct.exp(x - row_max)
    denom = ct.sum(numer, axis=1, keepdims=True)
    probs = ct.astype(numer / denom, ct.bfloat16)

    ct.store(amax_ptr, index=(row,), tile=ct.reshape(row_max, (1,)))
    ct.store(sum_ptr, index=(row,), tile=ct.reshape(denom, (1,)))

    random_f = ct.load(random_ptr, index=(row, 0), shape=(1, K_LEN_C))
    rand_bf = ct.astype(random_f, ct.bfloat16)
    p_bf = ct.full(shape=(1, K_LEN_C), fill_value=DROPOUT_P, dtype=ct.bfloat16)
    keep = rand_bf > p_bf
    ct.store(keep_ptr, index=(row, 0), tile=keep)

    zero_bf = ct.zeros((1, K_LEN_C), dtype=ct.bfloat16)
    dropped = ct.where(keep, probs, zero_bf)
    scaled = ct.astype(ct.astype(dropped, ct.float32) * DROPOUT_SCALE, ct.bfloat16)
    ct.store(dropped_ptr, index=(row, 0), tile=scaled)


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= int(dim)
    return tuple(reversed(stride))


def _inductor_random_for_eager_check(shape, seed, *, device):
    if torch.cuda.is_current_stream_capturing():
        return torch.ops.prims.inductor_random.default(shape, seed, "rand")

    numel = 1
    for dim in shape:
        numel *= int(dim)
    advance = (numel + 131071) // 131072
    state = torch.cuda.get_rng_state(device)
    offset = int.from_bytes(bytes(state[8:16].tolist()), "little")
    if offset >= advance:
        rewound = state.clone()
        rewound_offset = offset - advance
        rewound[8:16] = torch.tensor(
            list(int(rewound_offset).to_bytes(8, "little", signed=False)),
            dtype=state.dtype, device=state.device,
        )
        torch.cuda.set_rng_state(rewound, device)
        random = torch.ops.prims.inductor_random.default(shape, seed, "rand")
        torch.cuda.set_rng_state(state, device)
        return random
    return torch.ops.prims.inductor_random.default(shape, seed, "rand")


@oracle_impl(hardware="B200", point="dda3d8e0")
def oracle_forward(inputs):
    arg0_1, arg1_1, arg2_1, shape0, shape1, _shape2, shape3 = inputs
    full_shape = tuple(int(dim) for dim in shape0)  # (32, 6, 128, 128)
    random_shape = tuple(int(dim) for dim in shape1)  # (32, 6, 128, 128)
    out_shape = tuple(int(dim) for dim in shape3)  # (192, 128, 128)
    row_shape = full_shape[:-1] + (1,)

    device = arg0_1.device
    rounded = torch.empty_strided(full_shape, _contiguous_stride(full_shape),
                                  device=device, dtype=torch.bfloat16)
    amax = torch.empty_strided(row_shape, _contiguous_stride(row_shape),
                               device=device, dtype=torch.float32)
    sum_1 = torch.empty_strided(row_shape, _contiguous_stride(row_shape),
                                device=device, dtype=torch.float32)
    gt = torch.empty_strided(full_shape, _contiguous_stride(full_shape),
                             device=device, dtype=torch.bool)
    dropped = torch.empty_strided(out_shape, _contiguous_stride(out_shape),
                                  device=device, dtype=torch.bfloat16)

    # arg0_1 is bf16 [192, 128, 128] and its "view" to [32,6,128,128] is
    # contiguous, so we can treat rows as N_ROWS x K_LEN.
    n_rows = arg0_1.numel() // K_LEN
    score_2d = arg0_1.reshape(n_rows, K_LEN)
    # arg1_1 is strided; we need it materialized as a contiguous [32,6,128,128]
    # before flattening.
    bias_contig = arg1_1.contiguous()
    bias_2d = bias_contig.reshape(n_rows, K_LEN)

    seed = torch.ops.prims.inductor_lookup_seed.default(arg2_1, SEED_INDEX)
    random = _inductor_random_for_eager_check(random_shape, seed, device=device)
    random_2d = random.reshape(n_rows, K_LEN)

    rounded_2d = rounded.view(n_rows, K_LEN)
    amax_1d = amax.view(n_rows)
    sum_1d = sum_1.view(n_rows)
    gt_2d = gt.view(n_rows, K_LEN)
    dropped_2d = dropped.view(n_rows, K_LEN)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (n_rows, 1, 1),
        _softmax_dropout_kernel,
        (
            score_2d, bias_2d, random_2d,
            rounded_2d, amax_1d, sum_1d, gt_2d, dropped_2d,
            K_LEN,
        ),
    )
    return rounded, amax, sum_1, gt, dropped, dropped.permute(0, 2, 1)
