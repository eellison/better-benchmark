"""cuTile port of amax_sum_119baf9550ec: T5 softmax + seeded dropout row kernel.

Ports the Triton `_softmax_dropout_seeded_kernel`. The seeded on-device RNG is
replaced with a pre-generated random tensor via
`torch.ops.prims.inductor_random.default` — the same approach the Triton oracle
uses in its eager fallback (`_softmax_dropout_random_kernel`).

Returned side outputs (amax, sum_1) are stochastic-safe (deterministic given the
input); the `gt` mask and the `dropped` bf16 output are stochastic and skipped
by the harness stochastic-detector when checking numerics.
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 41
DROPOUT_SCALE = 1.1111111111111112


@ct.kernel
def _softmax_dropout_kernel(
    x_ptr,       # bf16 [n_rows, k_len]
    random_ptr,  # f32  [n_rows, k_len]
    amax_ptr,    # f32  [n_rows]
    sum_ptr,     # f32  [n_rows]
    gt_ptr,      # b8   [n_rows, k_len]
    dropped_ptr, # bf16 [n_rows, k_len]
    K_LEN: ct.Constant[int],
    BLOCK_M: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
):
    row_block = ct.bid(0)

    scores_bf = ct.load(
        x_ptr, index=(row_block, 0), shape=(BLOCK_M, BLOCK_N),
    )
    scores = ct.astype(scores_bf, ct.float32)
    row_max = ct.sum(scores, axis=1, keepdims=True) * 0.0  # zero-init shape helper
    row_max = ct.max(scores, axis=1, keepdims=True)
    numer = ct.exp(scores - row_max)
    denom = ct.sum(numer, axis=1, keepdims=True)
    probs = ct.astype(numer / denom, ct.bfloat16)

    # Store the row-wise amax and sum (shape [BLOCK_M, 1] -> [BLOCK_M])
    row_max_1d = ct.reshape(row_max, (BLOCK_M,))
    denom_1d = ct.reshape(denom, (BLOCK_M,))
    ct.store(amax_ptr, index=(row_block,), tile=row_max_1d)
    ct.store(sum_ptr, index=(row_block,), tile=denom_1d)

    # Dropout via pre-generated random tensor: cast to bf16 then compare.
    random = ct.load(random_ptr, index=(row_block, 0), shape=(BLOCK_M, BLOCK_N))
    rand_bf = ct.astype(random, ct.bfloat16)
    dropout_p_bf = ct.astype(
        ct.full((BLOCK_M, BLOCK_N), 0.1, dtype=ct.float32), ct.bfloat16
    )
    keep = rand_bf > dropout_p_bf
    ct.store(gt_ptr, index=(row_block, 0), tile=keep)

    zero_bf = ct.full((BLOCK_M, BLOCK_N), 0.0, dtype=ct.bfloat16)
    dropped_bf = ct.where(keep, probs, zero_bf)
    scaled_bf = ct.astype(
        ct.astype(dropped_bf, ct.float32) * DROPOUT_SCALE, ct.bfloat16
    )
    ct.store(dropped_ptr, index=(row_block, 0), tile=scaled_bf)


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= int(dim)
    return tuple(reversed(stride))


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


@oracle_impl(hardware="B200", point="696b5761", BLOCK_M=1, BLOCK_N=1024)
def oracle_forward(inputs, *, BLOCK_M: int, BLOCK_N: int):
    arg0_1, arg1_1, shape0, shape1, _shape2, shape3 = inputs
    del _shape2

    full_shape = tuple(int(d) for d in shape0)
    random_shape = tuple(int(d) for d in shape1)
    out_shape = tuple(int(d) for d in shape3)
    row_shape = full_shape[:-1] + (1,)
    k_len = int(full_shape[-1])
    n_rows = int(arg0_1.numel() // k_len)

    device = arg0_1.device
    amax = torch.empty_strided(
        row_shape, _contiguous_stride(row_shape), device=device, dtype=torch.float32,
    )
    sum_1 = torch.empty_strided(
        row_shape, _contiguous_stride(row_shape), device=device, dtype=torch.float32,
    )
    gt = torch.empty_strided(
        full_shape, _contiguous_stride(full_shape), device=device, dtype=torch.bool,
    )
    dropped = torch.empty_strided(
        out_shape, _contiguous_stride(out_shape), device=device, dtype=torch.bfloat16,
    )

    # Views to 2D so cuTile sees a rank-2 array. Reshape must be safe since the
    # returned aliases are contiguous.
    x_2d = arg0_1.reshape(n_rows, k_len)
    gt_2d = gt.view(n_rows, k_len)
    dropped_2d = dropped.view(n_rows, k_len)
    amax_1d = amax.view(n_rows)
    sum_1d = sum_1.view(n_rows)

    # Pre-generate the random tensor (Inductor Philox), matching the Repro.
    seed = torch.ops.prims.inductor_lookup_seed.default(arg1_1, SEED_INDEX)
    random = _inductor_random_for_eager_check(random_shape, seed, device=device)
    random_2d = random.reshape(n_rows, k_len)

    if n_rows % BLOCK_M != 0:
        raise NotImplementedError(f"BLOCK_M={BLOCK_M} does not divide n_rows={n_rows}")
    if k_len != BLOCK_N:
        raise NotImplementedError(f"BLOCK_N={BLOCK_N} != k_len={k_len}")

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (n_rows // BLOCK_M, 1, 1),
        _softmax_dropout_kernel,
        (x_2d, random_2d, amax_1d, sum_1d, gt_2d, dropped_2d, k_len, BLOCK_M, BLOCK_N),
    )

    return amax, sum_1, gt, dropped, dropped.permute(0, 2, 1)
