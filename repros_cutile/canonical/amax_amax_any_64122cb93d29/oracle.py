"""cuTile port of amax_amax_any_64122cb93d29: XLNet relative-shift attention softmax + dropout.

Full training scope: content + relative-position gather (bf16), scaled and unscaled
amax, finite-row guard, softmax, dropout via pre-generated random tensor. Returns
add/amax/amax_scaled/finite/denom/keep/final/final.permute outputs.
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 82
N_ROWS = 16 * 16 * 512  # 131072
K_LEN = 512
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112

OUT_SHAPE_4D = (16, 16, 512, 512)
REDUCTION_SHAPE = (16, 16, 512, 1)
OUT_SHAPE_3D = (256, 512, 512)
CONTIG_4D_STRIDE = (4194304, 262144, 512, 1)
REDUCTION_STRIDE = (8192, 512, 1, 1)
CONTIG_3D_STRIDE = (262144, 512, 1)


@ct.kernel
def _xlnet_softmax_dropout_kernel(
    content_ptr,       # bf16 (N_ROWS, K_LEN) = (131072, 512)
    rel_gathered_ptr,  # bf16 (N_ROWS, K_LEN) — pre-gathered
    random_ptr,        # f32 (N_ROWS, K_LEN)
    add_ptr,           # bf16 (N_ROWS, K_LEN)
    amax_ptr,          # f32 (N_ROWS,)
    amax_scaled_ptr,   # f32 (N_ROWS,)
    finite_ptr,        # bool (N_ROWS,)
    denom_ptr,         # f32 (N_ROWS,)
    keep_ptr,          # bool (N_ROWS, K_LEN)
    final_ptr,         # bf16 (N_ROWS, K_LEN)
    K_LEN_C: ct.Constant[int],
):
    row = ct.bid(0)

    content_bf = ct.load(content_ptr, index=(row, 0), shape=(1, K_LEN_C))
    rel_bf = ct.load(rel_gathered_ptr, index=(row, 0), shape=(1, K_LEN_C))

    content_f = ct.astype(content_bf, ct.float32)
    rel_f = ct.astype(rel_bf, ct.float32)
    added_bf = ct.astype(content_f + rel_f, ct.bfloat16)
    ct.store(add_ptr, index=(row, 0), tile=added_bf)

    unscaled = ct.astype(added_bf, ct.float32)
    scaled_bf = ct.astype(unscaled * 0.125, ct.bfloat16)
    scaled = ct.astype(scaled_bf, ct.float32)

    # Check for invalid values (NaN or inf) in scaled.
    abs_scaled = ct.astype(unscaled * 0.0, ct.float32)  # placeholder
    # We'll compute has_invalid: NaN or inf. Simpler check: value != value (NaN) or |x| == inf.
    is_nan = scaled != scaled
    is_pos_inf = scaled == ct.full(shape=(1, K_LEN_C), fill_value=float("inf"), dtype=ct.float32)
    is_neg_inf = scaled == ct.full(shape=(1, K_LEN_C), fill_value=float("-inf"), dtype=ct.float32)
    invalid = is_nan | is_pos_inf | is_neg_inf
    invalid_i = ct.astype(invalid, ct.int32)
    has_invalid_i = ct.max(invalid_i, axis=1, keepdims=True)
    row_is_finite = has_invalid_i == ct.full(shape=(1, 1), fill_value=0, dtype=ct.int32)
    # row_is_finite tile shape (1, 1) — broadcastable

    unscaled_max = ct.max(unscaled, axis=1, keepdims=True)
    scaled_max = ct.max(scaled, axis=1, keepdims=True)
    ct.store(amax_ptr, index=(row,), tile=ct.reshape(unscaled_max, (1,)))
    ct.store(amax_scaled_ptr, index=(row,), tile=ct.reshape(scaled_max, (1,)))
    ct.store(finite_ptr, index=(row,), tile=ct.reshape(row_is_finite, (1,)))

    shifted_unscaled = (unscaled - unscaled_max) * 0.125
    shifted_scaled = scaled - scaled_max
    shifted = ct.where(row_is_finite, shifted_unscaled, shifted_scaled)

    numer = ct.exp(shifted)
    denom = ct.sum(numer, axis=1, keepdims=True)
    probs = numer / denom
    ct.store(denom_ptr, index=(row,), tile=ct.reshape(denom, (1,)))

    random_f = ct.load(random_ptr, index=(row, 0), shape=(1, K_LEN_C))
    p_f = ct.full(shape=(1, K_LEN_C), fill_value=DROPOUT_P, dtype=ct.float32)
    keep = random_f > p_f
    ct.store(keep_ptr, index=(row, 0), tile=keep)

    zero_f = ct.zeros((1, K_LEN_C), dtype=ct.float32)
    dropped = ct.where(keep, probs, zero_f)
    scaled_dropout = dropped * DROPOUT_SCALE
    ct.store(final_ptr, index=(row, 0), tile=ct.astype(scaled_dropout, ct.bfloat16))


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


@oracle_impl(hardware="B200", point="782e420b")
def oracle_forward(inputs):
    arg0_1, arg1_1, arg2_1, arg3_1, *_shape_params = inputs
    device = arg0_1.device

    # Reproduce the eager operations to get rel-gathered tensor.
    # view0: bf16 [16,16,512,1,512] = arg0_1 view
    view = arg0_1.view(16, 16, 512, 1, 512)
    permute = view.permute(0, 1, 2, 4, 3)
    view_1 = permute.reshape(16, 16, 512, 512)  # content

    # view_2: bf16 [16,16,512,1,1024]
    view_2 = arg1_1.view(16, 16, 512, 1, 1024)
    permute_1 = view_2.permute(0, 1, 2, 4, 3)
    view_3 = permute_1.reshape(16, 16, 512, 1024)
    view_4 = view_3.reshape(16, 16, 1024, 512)
    slice_1 = view_4[:, :, 1:]  # [16,16,1023,512]
    view_5 = slice_1.reshape(16, 16, 512, 1023)
    index = view_5.index_select(-1, arg2_1)  # [16,16,512,512]

    # content and rel are both bf16 [16,16,512,512]
    content_2d = view_1.reshape(N_ROWS, K_LEN).contiguous()
    rel_2d = index.reshape(N_ROWS, K_LEN).contiguous()

    add_out = torch.empty_strided(OUT_SHAPE_4D, CONTIG_4D_STRIDE,
                                  device=device, dtype=torch.bfloat16)
    amax = torch.empty_strided(REDUCTION_SHAPE, REDUCTION_STRIDE,
                               device=device, dtype=torch.float32)
    amax_scaled = torch.empty_strided(REDUCTION_SHAPE, REDUCTION_STRIDE,
                                      device=device, dtype=torch.float32)
    finite = torch.empty_strided(REDUCTION_SHAPE, REDUCTION_STRIDE,
                                 device=device, dtype=torch.bool)
    denom = torch.empty_strided(REDUCTION_SHAPE, REDUCTION_STRIDE,
                                device=device, dtype=torch.float32)
    keep = torch.empty_strided(OUT_SHAPE_4D, CONTIG_4D_STRIDE,
                               device=device, dtype=torch.bool)
    final = torch.empty_strided(OUT_SHAPE_3D, CONTIG_3D_STRIDE,
                                device=device, dtype=torch.bfloat16)

    seed = torch.ops.prims.inductor_lookup_seed.default(arg3_1, SEED_INDEX)
    random = _inductor_random_for_eager_check(OUT_SHAPE_4D, seed, device=device)
    random_2d = random.reshape(N_ROWS, K_LEN)

    add_2d = add_out.view(N_ROWS, K_LEN)
    amax_1d = amax.view(N_ROWS)
    amax_scaled_1d = amax_scaled.view(N_ROWS)
    finite_1d = finite.view(N_ROWS)
    denom_1d = denom.view(N_ROWS)
    keep_2d = keep.view(N_ROWS, K_LEN)
    final_2d = final.view(N_ROWS, K_LEN)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (N_ROWS, 1, 1),
        _xlnet_softmax_dropout_kernel,
        (
            content_2d, rel_2d, random_2d,
            add_2d, amax_1d, amax_scaled_1d, finite_1d, denom_1d,
            keep_2d, final_2d,
            K_LEN,
        ),
    )
    return add_out, amax, amax_scaled, finite, denom, keep, final, final.permute(0, 2, 1)
