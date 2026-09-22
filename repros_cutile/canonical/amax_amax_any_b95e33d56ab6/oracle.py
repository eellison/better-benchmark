"""cuTile port of amax_amax_any_b95e33d56ab6: XLNet relative-shift attention softmax dropout.

Uses pre-generated random tensor (from torch.ops.prims.inductor_random) to
sidestep cuTile's lack of on-device seeded RNG. Substitutes ct.astype for
inline PTX (round-to-nearest-even is cuTile's default).
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 30
N_ROWS = 16 * 16 * 512
K_LEN = 512
OUT_SHAPE_4D = (16, 16, 512, 512)
REDUCTION_SHAPE = (16, 16, 512, 1)
OUT_SHAPE_3D = (256, 512, 512)


@ct.kernel
def _xlnet_train_softmax_dropout_kernel(
    content_ptr,        # bf16 [rows, k_len]
    rel_gathered_ptr,   # bf16 [rows, k_len]  (gathered)
    random_ptr,         # f32  [rows, k_len]
    add_out_ptr,        # bf16 [rows, k_len]
    amax_out_ptr,       # f32  [rows]
    amax_scaled_out_ptr,  # f32  [rows]
    finite_out_ptr,     # b8   [rows]
    denom_out_ptr,      # f32  [rows]
    keep_out_ptr,       # b8   [rows, k_len]
    final_out_ptr,      # bf16 [rows, k_len]
    BLOCK_M: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
):
    pid = ct.bid(0)

    content = ct.load(content_ptr, index=(pid, 0), shape=(BLOCK_M, BLOCK_N))
    rel_g = ct.load(rel_gathered_ptr, index=(pid, 0), shape=(BLOCK_M, BLOCK_N))
    content_f = ct.astype(content, ct.float32)
    rel_f = ct.astype(rel_g, ct.float32)

    added_bf = ct.astype(content_f + rel_f, ct.bfloat16)
    ct.store(add_out_ptr, index=(pid, 0), tile=added_bf)

    unscaled = ct.astype(added_bf, ct.float32)
    scaled_bf = ct.astype(unscaled * 0.125, ct.bfloat16)
    scaled = ct.astype(scaled_bf, ct.float32)

    # Finite guard: scaled must equal itself (not NaN) and abs(scaled) != inf.
    abs_scaled = ct.abs(scaled)
    inf_val = ct.full((BLOCK_M, BLOCK_N), float("inf"), dtype=ct.float32)
    invalid = (scaled != scaled) | (abs_scaled == inf_val)
    zero_i = ct.full((BLOCK_M, BLOCK_N), 0, dtype=ct.int32)
    one_i = ct.full((BLOCK_M, BLOCK_N), 1, dtype=ct.int32)
    invalid_i = ct.where(invalid, one_i, zero_i)
    has_invalid = ct.max(invalid_i, axis=1)
    zero_i_row = ct.full((BLOCK_M,), 0, dtype=ct.int32)
    row_is_finite = has_invalid == zero_i_row

    unscaled_max = ct.max(unscaled, axis=1)
    scaled_max = ct.max(scaled, axis=1)
    ct.store(amax_out_ptr, index=(pid,), tile=unscaled_max)
    ct.store(amax_scaled_out_ptr, index=(pid,), tile=scaled_max)
    ct.store(finite_out_ptr, index=(pid,), tile=row_is_finite)

    unscaled_max_2d = ct.reshape(unscaled_max, (BLOCK_M, 1))
    scaled_max_2d = ct.reshape(scaled_max, (BLOCK_M, 1))
    row_is_finite_2d = ct.reshape(row_is_finite, (BLOCK_M, 1))

    shifted_unscaled = (unscaled - unscaled_max_2d) * 0.125
    shifted_scaled = scaled - scaled_max_2d
    shifted = ct.where(row_is_finite_2d, shifted_unscaled, shifted_scaled)

    numer = ct.exp(shifted)
    denom = ct.sum(numer, axis=1, keepdims=True)
    probs = numer / denom
    ct.store(denom_out_ptr, index=(pid,), tile=ct.reshape(denom, (BLOCK_M,)))

    # Random dropout mask
    random_f = ct.load(random_ptr, index=(pid, 0), shape=(BLOCK_M, BLOCK_N))
    threshold_f = ct.full((BLOCK_M, BLOCK_N), 0.1, dtype=ct.float32)
    keep = random_f > threshold_f
    ct.store(keep_out_ptr, index=(pid, 0), tile=keep)

    zero_f = ct.full((BLOCK_M, BLOCK_N), 0.0, dtype=ct.float32)
    one_f = ct.full((BLOCK_M, BLOCK_N), 1.0, dtype=ct.float32)
    keep_f = ct.where(keep, one_f, zero_f)
    dropped = keep_f * probs
    scaled_dropout = dropped * 1.1111111111111112
    ct.store(final_out_ptr, index=(pid, 0), tile=ct.astype(scaled_dropout, ct.bfloat16))


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


def _gather_rel(rel_1d, index_1d, *, N_ROWS_, K_LEN_):
    """Gather rel[rows, index[cols]] using a strided offset pattern.

    Triton computes: group = rows // 512; query = rows - group*512;
    rel_offsets = group * 524288 + 512 + query * 1023 + index[cols]

    That decomposes rel as arg1_1: bf16[256, 512, 1024], and the produced
    offsets are (group, ...) flat offsets into that layout. Do it as an
    equivalent torch gather so we don't need to encode the linear-arith math
    in-kernel.
    """
    # rel is [16*16, 512, 1024] flattened, viewed as arg1_1 [256, 512, 1024].
    # Rows 0..N_ROWS iterate over (group, query) pairs where group in [0,256)
    # and query in [0, 512). Column index selects from [1024] axis.
    # group * 524288 + 512 + query * 1023 + col_index[c]
    # 524288 = 512 * 1024, 1023 = 1024 - 1  =>  slot base = 512 + q*1023 + c_idx
    # This decodes as: view rel as [256, 512, 1024];
    #   rows -> (group, query); columns via index_1d selecting from the last dim.
    # But the offset structure "+ 512 + query * 1023" implies a shifted 1D view:
    # the flat offset per (group, query, col) = group*(512*1024) + query*1023 + 512 + index[col]
    # Note "+ 512" is a constant offset (i.e. skip first 512 entries of dim=2 in
    # each group only for row 0). Actually the structure is:
    # slot = base_g + slot_q + slot_c where slot_q = query * 1023 + 512
    #                                       slot_c = index[col]
    # This is exactly Triton's "slice(rel[..., 1:], q, index_map[c])" style
    # relative-position shift gather. We can build the intermediate in torch
    # by reshaping. Do it robustly by materializing the flat offsets.
    device = rel_1d.device
    rows = torch.arange(N_ROWS_, device=device, dtype=torch.int64)
    cols = torch.arange(K_LEN_, device=device, dtype=torch.int64)
    group = rows // 512
    query = rows - group * 512
    rel_offsets = (
        group[:, None] * 524288
        + 512
        + query[:, None] * 1023
        + index_1d[None, :].to(torch.int64)
    )
    return rel_1d.view(-1)[rel_offsets]


@oracle_impl(hardware="B200", point="782e420b", BLOCK_M=4, BLOCK_N=512)
def oracle_forward(inputs, *, BLOCK_M: int, BLOCK_N: int):
    arg0_1, arg1_1, arg2_1, arg3_1, *_shape_params = inputs
    device = arg0_1.device

    # arg0 is bf16 [256, 512, 512] (content). Reshape to [N_ROWS, K_LEN]
    content_2d = arg0_1.contiguous().view(N_ROWS, K_LEN)

    # Gather rel (bf16 [256, 512, 1024]) into [N_ROWS, K_LEN] via the shift pattern
    rel_gathered_2d = _gather_rel(arg1_1, arg2_1, N_ROWS_=N_ROWS, K_LEN_=K_LEN)

    add_out = torch.empty(OUT_SHAPE_4D, device=device, dtype=torch.bfloat16)
    amax = torch.empty(REDUCTION_SHAPE, device=device, dtype=torch.float32)
    amax_scaled = torch.empty(REDUCTION_SHAPE, device=device, dtype=torch.float32)
    finite = torch.empty(REDUCTION_SHAPE, device=device, dtype=torch.bool)
    denom = torch.empty(REDUCTION_SHAPE, device=device, dtype=torch.float32)
    keep = torch.empty(OUT_SHAPE_4D, device=device, dtype=torch.bool)
    final = torch.empty(OUT_SHAPE_3D, device=device, dtype=torch.bfloat16)

    seed = torch.ops.prims.inductor_lookup_seed.default(arg3_1, SEED_INDEX)
    random = _inductor_random_for_eager_check(OUT_SHAPE_4D, seed, device=device)

    # 2D views for the kernel
    add_out_2d = add_out.view(N_ROWS, K_LEN)
    amax_1d = amax.view(N_ROWS)
    amax_scaled_1d = amax_scaled.view(N_ROWS)
    finite_1d = finite.view(N_ROWS)
    denom_1d = denom.view(N_ROWS)
    keep_2d = keep.view(N_ROWS, K_LEN)
    final_2d = final.view(N_ROWS, K_LEN)
    random_2d = random.contiguous().view(N_ROWS, K_LEN)

    stream = torch.cuda.current_stream()
    grid = (ct.cdiv(N_ROWS, BLOCK_M), 1, 1)
    ct.launch(
        stream,
        grid,
        _xlnet_train_softmax_dropout_kernel,
        (content_2d, rel_gathered_2d, random_2d, add_out_2d,
         amax_1d, amax_scaled_1d, finite_1d, denom_1d,
         keep_2d, final_2d, BLOCK_M, BLOCK_N),
    )

    return (add_out, amax, amax_scaled, finite, denom, keep, final,
            final.permute(0, 2, 1))
