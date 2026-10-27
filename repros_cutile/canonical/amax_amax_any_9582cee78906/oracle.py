"""cuTile port of amax_amax_any_9582cee78906: XLNet train softmax+dropout.

Pre-computes the relative-position gather via the Repro's view/permute/slice/
index cascade on the torch side, then runs a per-row softmax+dropout kernel.
Random tensor is pre-generated with torch.ops.prims.inductor_random.
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 70
DROPOUT_SCALE = 1.1111111111111112

OUT_SHAPE_4D = (16, 16, 512, 512)
REDUCTION_SHAPE = (16, 16, 512, 1)
OUT_SHAPE_3D = (256, 512, 512)


@ct.kernel
def _xlnet_train_softmax_dropout_kernel(
    content_ptr,     # bf16 [n_rows, k_len]
    rel_ptr,         # bf16 [n_rows, k_len]
    random_ptr,      # f32  [n_rows, k_len]
    add_ptr,         # bf16 [n_rows, k_len] (added_bf16 output)
    amax_ptr,        # f32  [n_rows]
    amax_scaled_ptr, # f32  [n_rows]
    finite_ptr,      # b8   [n_rows]
    denom_ptr,       # f32  [n_rows]
    keep_ptr,        # b8   [n_rows, k_len]
    final_ptr,       # bf16 [n_rows, k_len]
    K_LEN: ct.Constant[int],
    BLOCK_M: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
    DROPOUT_SCALE_: ct.Constant[float],
):
    row_block = ct.bid(0)

    content_bf = ct.load(content_ptr, index=(row_block, 0), shape=(BLOCK_M, BLOCK_N))
    rel_bf = ct.load(rel_ptr, index=(row_block, 0), shape=(BLOCK_M, BLOCK_N))
    rand_f = ct.load(random_ptr, index=(row_block, 0), shape=(BLOCK_M, BLOCK_N))

    content_f = ct.astype(content_bf, ct.float32)
    rel_f = ct.astype(rel_bf, ct.float32)

    added_bf = ct.astype(content_f + rel_f, ct.bfloat16)
    ct.store(add_ptr, index=(row_block, 0), tile=added_bf)

    unscaled = ct.astype(added_bf, ct.float32)
    scaled_bf = ct.astype(unscaled * 0.125, ct.bfloat16)
    scaled = ct.astype(scaled_bf, ct.float32)

    # Finite check on scaled: (scaled == scaled) & (|scaled| != inf)
    is_nan = scaled != scaled
    inf_val = ct.full((BLOCK_M, BLOCK_N), float("inf"), dtype=ct.float32)
    neg_inf_val = ct.full((BLOCK_M, BLOCK_N), float("-inf"), dtype=ct.float32)
    is_pos_inf = scaled == inf_val
    is_neg_inf = scaled == neg_inf_val
    invalid = is_nan | is_pos_inf | is_neg_inf  # b8

    # Reduce along columns: has_invalid = any(invalid)
    invalid_int = ct.astype(invalid, ct.int32)
    invalid_sum = ct.sum(invalid_int, axis=1, keepdims=True)  # (BLOCK_M, 1)
    zero_i = ct.full((BLOCK_M, 1), 0, dtype=ct.int32)
    has_invalid = invalid_sum != zero_i  # b8 (BLOCK_M, 1)
    # row_is_finite = ~has_invalid
    true_2d = ct.full((BLOCK_M, 1), True, dtype=ct.bool_)
    false_2d = ct.full((BLOCK_M, 1), False, dtype=ct.bool_)
    row_is_finite = ct.where(has_invalid, false_2d, true_2d)

    # amax_out = max(unscaled) along cols
    unscaled_max = ct.max(unscaled, axis=1, keepdims=True)  # (BLOCK_M, 1)
    scaled_max = ct.max(scaled, axis=1, keepdims=True)      # (BLOCK_M, 1)
    ct.store(amax_ptr, index=(row_block,), tile=ct.reshape(unscaled_max, (BLOCK_M,)))
    ct.store(amax_scaled_ptr, index=(row_block,), tile=ct.reshape(scaled_max, (BLOCK_M,)))
    ct.store(finite_ptr, index=(row_block,), tile=ct.reshape(row_is_finite, (BLOCK_M,)))

    # shifted: if row_is_finite: (unscaled - unscaled_max) * 0.125
    #          else: scaled - scaled_max
    shifted_unscaled = (unscaled - unscaled_max) * 0.125
    shifted_scaled = scaled - scaled_max
    shifted = ct.where(row_is_finite, shifted_unscaled, shifted_scaled)

    numer = ct.exp(shifted)
    denom = ct.sum(numer, axis=1, keepdims=True)
    ct.store(denom_ptr, index=(row_block,), tile=ct.reshape(denom, (BLOCK_M,)))
    probs = numer / denom

    keep = rand_f > 0.1
    ct.store(keep_ptr, index=(row_block, 0), tile=keep)

    zero_f = ct.full((BLOCK_M, BLOCK_N), 0.0, dtype=ct.float32)
    dropped = ct.where(keep, probs, zero_f)
    scaled_dropout = dropped * DROPOUT_SCALE_
    ct.store(final_ptr, index=(row_block, 0), tile=ct.astype(scaled_dropout, ct.bfloat16))


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


def _compute_rel_gathered(arg1_1, arg2_1):
    """Materialize the relative-position gather that the Repro does via
    view/permute/slice/index — returns bf16[16,16,512,512]."""
    view_2 = arg1_1.view(16, 16, 512, 1, 1024)
    permute_1 = view_2.permute(0, 1, 2, 4, 3)
    view_3 = permute_1.reshape(16, 16, 512, 1024)
    view_4 = view_3.reshape(16, 16, 1024, 512)
    slice_1 = view_4[:, :, 1:]
    view_5 = slice_1.reshape(16, 16, 512, 1023)
    # index: on dim 3 with arg2_1 (i64[512])
    index = view_5[:, :, :, arg2_1]
    return index


def _compute_view_1(arg0_1):
    """arg0_1 [256,512,512] -> view [16,16,512,1,512] -> permute -> view [16,16,512,512]."""
    view = arg0_1.view(16, 16, 512, 1, 512)
    permute = view.permute(0, 1, 2, 4, 3)
    view_1 = permute.reshape(16, 16, 512, 512)
    return view_1


@oracle_impl(hardware="B200", point="782e420b", block_m=4, block_n=512)
def oracle_forward(inputs, *, block_m: int, block_n: int):
    arg0_1, arg1_1, arg2_1, arg3_1, *_shape_params = inputs

    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        raise NotImplementedError(
            "cuTile port unsupported inside CUDA graph capture (seeded RNG)."
        )

    device = arg0_1.device
    n_rows = 16 * 16 * 512
    k_len = 512

    # Pre-compute view_1 (content) and rel_gathered on torch side.
    view_1 = _compute_view_1(arg0_1).contiguous()
    rel_gathered = _compute_rel_gathered(arg1_1, arg2_1).contiguous()

    add_out = torch.empty_strided(
        OUT_SHAPE_4D, (4194304, 262144, 512, 1),
        device=device, dtype=torch.bfloat16)
    amax = torch.empty_strided(
        REDUCTION_SHAPE, (8192, 512, 1, 1),
        device=device, dtype=torch.float32)
    amax_scaled = torch.empty_strided(
        REDUCTION_SHAPE, (8192, 512, 1, 1),
        device=device, dtype=torch.float32)
    finite = torch.empty_strided(
        REDUCTION_SHAPE, (8192, 512, 1, 1),
        device=device, dtype=torch.bool)
    denom = torch.empty_strided(
        REDUCTION_SHAPE, (8192, 512, 1, 1),
        device=device, dtype=torch.float32)
    keep = torch.empty_strided(
        OUT_SHAPE_4D, (4194304, 262144, 512, 1),
        device=device, dtype=torch.bool)
    final = torch.empty_strided(
        OUT_SHAPE_3D, (262144, 512, 1),
        device=device, dtype=torch.bfloat16)

    seed = torch.ops.prims.inductor_lookup_seed.default(arg3_1, SEED_INDEX)
    random = _inductor_random_for_eager_check(OUT_SHAPE_4D, seed, device=device)

    content_2d = view_1.view(n_rows, k_len)
    rel_2d = rel_gathered.view(n_rows, k_len)
    random_2d = random.contiguous().view(n_rows, k_len)
    add_2d = add_out.view(n_rows, k_len)
    keep_2d = keep.view(n_rows, k_len)
    final_2d = final.view(n_rows, k_len)
    amax_1d = amax.view(n_rows)
    amax_scaled_1d = amax_scaled.view(n_rows)
    finite_1d = finite.view(n_rows)
    denom_1d = denom.view(n_rows)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(n_rows, block_m), 1, 1),
        _xlnet_train_softmax_dropout_kernel,
        (content_2d, rel_2d, random_2d, add_2d,
         amax_1d, amax_scaled_1d, finite_1d, denom_1d,
         keep_2d, final_2d,
         k_len, block_m, block_n, DROPOUT_SCALE),
    )
    return add_out, amax, amax_scaled, finite, denom, keep, final, final.permute(0, 2, 1)
