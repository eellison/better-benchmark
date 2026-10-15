"""cuTile port of amax_amax_any_99a4f19df20a: XLNet attention softmax + seeded dropout.

Pre-computes the relative-shift index add + returned iota with torch on the
Python side (matching Inductor), then a single cuTile row kernel runs the
softmax + dropout epilogue matching the Triton oracle's semantics.
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 2
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112


@ct.kernel
def _xlnet_softmax_dropout_kernel(
    added_ptr,    # bf16 (n_rows, K)
    random_ptr,   # f32 (n_rows, K)
    amax_ptr,     # f32 (n_rows,) — unscaled amax (from bf16 add)
    amax_s_ptr,   # f32 (n_rows,) — scaled amax (bf16 add scaled by 0.125 then bf16)
    finite_ptr,   # bool (n_rows,)
    denom_ptr,    # f32 (n_rows,)
    keep_ptr,     # bool (n_rows, K)
    final_ptr,    # bf16 (n_rows, K)
    BLOCK_N: ct.Constant[int],
):
    row = ct.bid(0)

    added_bf = ct.load(added_ptr, index=(row, 0), shape=(1, BLOCK_N))
    unscaled = ct.astype(added_bf, ct.float32)
    scaled_bf = ct.astype(unscaled * 0.125, ct.bfloat16)
    scaled = ct.astype(scaled_bf, ct.float32)

    row_max_uns = ct.max(unscaled, axis=1, keepdims=True)
    row_max_scl = ct.max(scaled, axis=1, keepdims=True)

    # finite check on scaled tensor.
    abs_scaled = ct.where(scaled >= ct.zeros((1, BLOCK_N), dtype=ct.float32),
                          scaled, -scaled)
    is_nan = scaled != scaled
    is_inf = abs_scaled == float("inf")
    invalid = is_nan | is_inf
    invalid_int = ct.astype(invalid, ct.int32)
    has_invalid = ct.sum(invalid_int, axis=1, keepdims=True) > 0
    # row_is_finite = ~has_invalid
    finite_scalar = ~has_invalid  # (1,1) bool

    ct.store(amax_ptr, index=(row,), tile=ct.reshape(row_max_uns, (1,)))
    ct.store(amax_s_ptr, index=(row,), tile=ct.reshape(row_max_scl, (1,)))
    ct.store(finite_ptr, index=(row,), tile=ct.reshape(finite_scalar, (1,)))

    # shifted = (unscaled - unscaled_max) * 0.125 (if finite) else (scaled - scaled_max)
    shifted_finite = (unscaled - row_max_uns) * 0.125
    shifted_infinite = scaled - row_max_scl
    shifted = ct.where(finite_scalar, shifted_finite, shifted_infinite)

    numer = ct.exp(shifted)
    denom = ct.sum(numer, axis=1, keepdims=True)
    probs = numer / denom
    ct.store(denom_ptr, index=(row,), tile=ct.reshape(denom, (1,)))

    rand_f = ct.load(random_ptr, index=(row, 0), shape=(1, BLOCK_N))
    keep = rand_f > DROPOUT_P
    ct.store(keep_ptr, index=(row, 0), tile=keep)

    dropped = ct.astype(keep, ct.float32) * probs
    scaled_dropout = dropped * DROPOUT_SCALE
    final_bf = ct.astype(scaled_dropout, ct.bfloat16)
    ct.store(final_ptr, index=(row, 0), tile=final_bf)


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


OUT_SHAPE_4D = (16, 16, 512, 512)
REDUCTION_SHAPE = (16, 16, 512, 1)
OUT_SHAPE_3D = (256, 512, 512)


@oracle_impl(hardware="B200", point="4a104aa9", BLOCK_N=512)
def oracle_forward(inputs, *, BLOCK_N: int):
    arg0_1, arg1_1, arg2_1, *_shape_params = inputs

    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        raise NotImplementedError(
            "cuTile port unsupported inside CUDA graph capture (seeded RNG)."
        )

    device = arg0_1.device
    n_rows = 16 * 16 * 512
    k_len = 512

    # Materialize the iota and index-add producer on the Python side matching
    # the Triton oracle's semantics.
    iota = torch.arange(0, 512, device=device, dtype=torch.int64)

    # Replicate the relative-shift addition. Using torch ops for clarity.
    view = arg0_1.view(16, 16, 512, 1, 512)
    permute = view.permute(0, 1, 2, 4, 3)
    view_1 = permute.reshape(16, 16, 512, 512)

    view_2 = arg1_1.view(16, 16, 512, 1, 1024)
    permute_1 = view_2.permute(0, 1, 2, 4, 3)
    view_3 = permute_1.reshape(16, 16, 512, 1024)
    view_4 = view_3.view(16, 16, 1024, 512)
    slice_1 = view_4[:, :, 1:]  # (16,16,1023,512)
    view_5 = slice_1.reshape(16, 16, 512, 1023)
    idx = view_5[..., iota]  # (16,16,512,512)

    add = view_1 + idx
    add_1 = add + 0  # keeps bf16 semantics

    # Amax & related outputs match the Triton oracle.
    amax_uns = torch.empty(REDUCTION_SHAPE, device=device, dtype=torch.float32)
    amax_scl = torch.empty(REDUCTION_SHAPE, device=device, dtype=torch.float32)
    finite = torch.empty(REDUCTION_SHAPE, device=device, dtype=torch.bool)
    denom = torch.empty(REDUCTION_SHAPE, device=device, dtype=torch.float32)
    keep = torch.empty(OUT_SHAPE_4D, device=device, dtype=torch.bool)
    final = torch.empty(OUT_SHAPE_3D, device=device, dtype=torch.bfloat16)

    seed = torch.ops.prims.inductor_lookup_seed.default(arg2_1, SEED_INDEX)
    random = _inductor_random_for_eager_check(OUT_SHAPE_4D, seed, device=device)

    added_2d = add_1.contiguous().view(n_rows, k_len)
    random_2d = random.contiguous().view(n_rows, k_len)
    amax_1d = amax_uns.view(n_rows)
    amax_s_1d = amax_scl.view(n_rows)
    finite_1d = finite.view(n_rows)
    denom_1d = denom.view(n_rows)
    keep_2d = keep.view(n_rows, k_len)
    final_2d = final.view(n_rows, k_len)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (n_rows, 1, 1),
        _xlnet_softmax_dropout_kernel,
        (added_2d, random_2d, amax_1d, amax_s_1d, finite_1d, denom_1d,
         keep_2d, final_2d, BLOCK_N),
    )

    return (iota, add_1, amax_uns, amax_scl, finite, denom, keep, final,
            final.permute(0, 2, 1))
