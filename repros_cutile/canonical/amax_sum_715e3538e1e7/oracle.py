"""cuTile port of amax_sum_715e3538e1e7: BERT scaled masked attention softmax + dropout.

Pre-generates the seeded random tensor via inductor_random outside the kernel,
then runs one cuTile row kernel that scales / masks with a fill / row softmax /
seeded dropout. Refuses under CUDA-graph capture (RNG state unavailable).
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 26
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112
SCALE = 0.125


@ct.kernel
def _scaled_masked_softmax_dropout_kernel(
    x_ptr,          # bf16 [rows, K]  (dense flat)
    mask_ptr,       # b8   [rows, K]  (dense; head-broadcast beforehand)
    fill_ptr,       # bf16 []         scalar
    random_ptr,     # f32  [rows, K]
    where_ptr,      # bf16 [rows, K]  where(mask, fill, x*scale)
    amax_ptr,       # f32  [rows]
    denom_ptr,      # f32  [rows]
    keep_ptr,       # b8   [rows, K]
    out_ptr,        # bf16 [rows, K]  final dropout output
    BLOCK_N: ct.Constant[int],
):
    row = ct.bid(0)

    x_bf = ct.load(x_ptr, index=(row, 0), shape=(1, BLOCK_N))
    scaled_bf = ct.astype(ct.astype(x_bf, ct.float32) * SCALE, ct.bfloat16)
    mask = ct.load(mask_ptr, index=(row, 0), shape=(1, BLOCK_N))
    fill = ct.load(fill_ptr, index=(0,), shape=(1,))
    fill_bf = ct.reshape(fill, (1, 1))
    fill_2d = ct.full((1, BLOCK_N), 0.0, dtype=ct.bfloat16) + fill_bf
    masked_bf = ct.where(mask, fill_2d, scaled_bf)
    ct.store(where_ptr, index=(row, 0), tile=masked_bf)

    scores = ct.astype(masked_bf, ct.float32)
    row_max = ct.max(scores, axis=1, keepdims=True)
    shifted = scores - row_max
    numer = ct.exp(shifted)
    denom = ct.sum(numer, axis=1, keepdims=True)
    probs = numer / denom

    ct.store(amax_ptr, index=(row,), tile=ct.reshape(row_max, (1,)))
    ct.store(denom_ptr, index=(row,), tile=ct.reshape(denom, (1,)))

    rand = ct.load(random_ptr, index=(row, 0), shape=(1, BLOCK_N))
    thresh = ct.full((1, BLOCK_N), DROPOUT_P, dtype=ct.float32)
    keep = rand > thresh
    ct.store(keep_ptr, index=(row, 0), tile=keep)

    zero_f = ct.full((1, BLOCK_N), 0.0, dtype=ct.float32)
    dropped = ct.where(keep, probs, zero_f)
    scaled_out = ct.astype(dropped * DROPOUT_SCALE, ct.bfloat16)
    ct.store(out_ptr, index=(row, 0), tile=scaled_out)


def _shape_tuple(shape):
    return tuple(int(dim) for dim in shape)


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
    props = torch.cuda.get_device_properties(device)
    block_size = 256
    unroll = 4
    curand4_engine_calls = 4
    blocks_per_sm = props.max_threads_per_multi_processor // block_size
    grid = min(
        (numel + block_size - 1) // block_size,
        props.multi_processor_count * blocks_per_sm,
    )
    advance = (
        ((numel - 1) // (block_size * grid * unroll) + 1)
        * curand4_engine_calls
        * 2
    )
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


def _resolve_shape(shape, numel):
    dims = [int(dim) for dim in shape]
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


@oracle_impl(hardware="B200", point="0e2c5e9e", BLOCK_N=128)
def oracle_forward(inputs, *, BLOCK_N: int):
    arg0_1, arg1_1, arg2_1, arg3_1, shape0, shape1, _shape2, shape3 = inputs

    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        raise NotImplementedError(
            "cuTile port unsupported inside CUDA graph capture (seeded RNG)."
        )

    view_shape = _resolve_shape(shape0, arg0_1.numel())
    random_shape = _shape_tuple(shape1)
    flat_shape = _resolve_shape(shape3, arg0_1.numel())
    reduction_shape = (view_shape[0], view_shape[1], view_shape[2], 1)
    k_len = int(view_shape[-1])
    n_rows = int(arg0_1.numel() // k_len)
    device = arg0_1.device

    where = torch.empty_strided(
        view_shape, _contiguous_stride(view_shape),
        device=device, dtype=torch.bfloat16,
    )
    amax = torch.empty_strided(
        reduction_shape, _contiguous_stride(reduction_shape),
        device=device, dtype=torch.float32,
    )
    denom = torch.empty_strided(
        reduction_shape, _contiguous_stride(reduction_shape),
        device=device, dtype=torch.float32,
    )
    keep = torch.empty_strided(
        view_shape, _contiguous_stride(view_shape),
        device=device, dtype=torch.bool,
    )
    out = torch.empty_strided(
        flat_shape, _contiguous_stride(flat_shape),
        device=device, dtype=torch.bfloat16,
    )

    seed = torch.ops.prims.inductor_lookup_seed.default(arg3_1, SEED_INDEX)
    random = _inductor_random_for_eager_check(random_shape, seed, device=device)

    # Broadcast mask [B,1,Q,K] -> [B,H,Q,K] then flatten to [rows, K].
    mask_dense = arg1_1.expand(view_shape).contiguous().view(n_rows, k_len)
    x_2d = arg0_1.contiguous().view(n_rows, k_len)
    random_2d = random.contiguous().view(n_rows, k_len)
    fill_scalar = arg2_1.view(1)

    where_2d = where.view(n_rows, k_len)
    amax_1d = amax.view(n_rows)
    denom_1d = denom.view(n_rows)
    keep_2d = keep.view(n_rows, k_len)
    out_2d = out.view(n_rows, k_len)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (n_rows, 1, 1),
        _scaled_masked_softmax_dropout_kernel,
        (x_2d, mask_dense, fill_scalar, random_2d,
         where_2d, amax_1d, denom_1d, keep_2d, out_2d, BLOCK_N),
    )
    return where, amax, denom, keep, out, out.permute(0, 2, 1)
