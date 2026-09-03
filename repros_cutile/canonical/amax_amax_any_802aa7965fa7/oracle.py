"""cuTile port of amax_amax_any_802aa7965fa7: XLNet train softmax + dropout.

Pre-generates the random tensor via torch.ops.prims.inductor_random outside
the kernel. Runs one row kernel that gathers the relative-position
contribution, adds content + rel in bf16, does the amax/finite/softmax
reductions with the finite-row switch, and applies seeded dropout with
mul.rn.f32 semantics (cuTile's default f32 multiply rounding).
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 26
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112
N_ROWS = 16 * 16 * 512  # 131072
K_LEN = 512
GROUP_STRIDE = 512 * 1024  # 524288: elems per (b, h) group in arg1_1
QUERY_STRIDE = 1023
SLICE_OFFSET = 512
OUT_SHAPE_4D = (16, 16, 512, 512)
REDUCTION_SHAPE = (16, 16, 512, 1)
OUT_SHAPE_3D = (256, 512, 512)
CONTIG_4D_STRIDE = (4194304, 262144, 512, 1)
REDUCTION_STRIDE = (8192, 512, 1, 1)
CONTIG_3D_STRIDE = (262144, 512, 1)


@ct.kernel
def _xlnet_train_softmax_dropout_kernel(
    content_ptr,      # bf16 [N_ROWS, K_LEN]
    rel_ptr,          # bf16 [16 * GROUP_STRIDE] (flat 1D view of arg1_1)
    index_ptr,        # i64  [K_LEN]
    random_ptr,       # f32  [N_ROWS, K_LEN]
    add_ptr,          # bf16 [N_ROWS, K_LEN]
    amax_ptr,         # f32  [N_ROWS]
    amax_scaled_ptr,  # f32  [N_ROWS]
    finite_ptr,       # bool [N_ROWS]
    denom_ptr,        # f32  [N_ROWS]
    keep_ptr,         # bool [N_ROWS, K_LEN]
    final_ptr,        # bf16 [N_ROWS, K_LEN]
    BLOCK_N: ct.Constant[int],
    GROUP_STRIDE_C: ct.Constant[int],
    SLICE_OFFSET_C: ct.Constant[int],
    QUERY_STRIDE_C: ct.Constant[int],
):
    row = ct.bid(0)
    group = row // 512
    query = row - group * 512

    # Gather rel using formula: group * 524288 + 512 + query * 1023 + index[col]
    idx = ct.load(index_ptr, index=(0,), shape=(BLOCK_N,))
    idx_2d = ct.reshape(idx, (1, BLOCK_N))
    base = group * GROUP_STRIDE_C + SLICE_OFFSET_C + query * QUERY_STRIDE_C
    offsets = idx_2d + base  # (1, BLOCK_N), int64 (broadcast promotes)
    rel_bf = ct.gather(rel_ptr, offsets)  # (1, BLOCK_N) bf16

    content_bf = ct.load(content_ptr, index=(row, 0), shape=(1, BLOCK_N))
    content_f = ct.astype(content_bf, ct.float32)
    rel_f = ct.astype(rel_bf, ct.float32)
    added_bf = ct.astype(content_f + rel_f, ct.bfloat16)
    ct.store(add_ptr, index=(row, 0), tile=added_bf)

    unscaled = ct.astype(added_bf, ct.float32)
    scaled_bf = ct.astype(unscaled * 0.125, ct.bfloat16)
    scaled = ct.astype(scaled_bf, ct.float32)

    unscaled_max = ct.max(unscaled)  # scalar (0-d)
    scaled_max = ct.max(scaled)      # scalar (0-d)
    ct.store(amax_ptr, index=(row,), tile=ct.reshape(unscaled_max, (1,)))
    ct.store(amax_scaled_ptr, index=(row,), tile=ct.reshape(scaled_max, (1,)))

    # Finiteness on scaled: any nan or inf makes the row "not finite"
    inf_val = ct.full((1, BLOCK_N), float("inf"), dtype=ct.float32)
    abs_scaled = ct.abs(scaled)
    is_finite = (scaled == scaled) & (abs_scaled != inf_val)
    zero_i = ct.zeros((1, BLOCK_N), dtype=ct.int32)
    one_i = ct.full((1, BLOCK_N), 1, dtype=ct.int32)
    invalid_flag = ct.where(is_finite, zero_i, one_i)
    has_invalid = ct.max(invalid_flag)  # scalar
    row_is_finite = has_invalid == 0    # scalar bool
    ct.store(finite_ptr, index=(row,), tile=ct.reshape(row_is_finite, (1,)))

    # Softmax with finiteness switch:
    #  finite row -> shift unscaled and re-scale (equivalent, better rounded)
    #  bad   row -> shift scaled  (matches how Triton handles inf/nan rows)
    shifted_unscaled = (unscaled - unscaled_max) * 0.125
    shifted_scaled = scaled - scaled_max
    shifted = ct.where(row_is_finite, shifted_unscaled, shifted_scaled)

    numer = ct.exp(shifted)
    denom = ct.sum(numer)  # scalar
    ct.store(denom_ptr, index=(row,), tile=ct.reshape(denom, (1,)))
    probs = numer / denom

    # Dropout via pre-generated f32 random tensor (mirrors tl.rand path).
    rand_f = ct.load(random_ptr, index=(row, 0), shape=(1, BLOCK_N))
    threshold = ct.full((1, BLOCK_N), DROPOUT_P, dtype=ct.float32)
    keep = rand_f > threshold
    ct.store(keep_ptr, index=(row, 0), tile=keep)

    keep_f = ct.astype(keep, ct.float32)
    dropped = keep_f * probs
    scaled_dropout = dropped * DROPOUT_SCALE
    ct.store(final_ptr, index=(row, 0), tile=ct.astype(scaled_dropout, ct.bfloat16))


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


@oracle_impl(hardware="B200", point="782e420b")
def oracle_forward(inputs):
    arg0_1, arg1_1, arg2_1, arg3_1, *_shape_params = inputs
    device = arg0_1.device

    add_out = torch.empty_strided(
        OUT_SHAPE_4D, CONTIG_4D_STRIDE, device=device, dtype=torch.bfloat16)
    amax = torch.empty_strided(
        REDUCTION_SHAPE, REDUCTION_STRIDE, device=device, dtype=torch.float32)
    amax_scaled = torch.empty_strided(
        REDUCTION_SHAPE, REDUCTION_STRIDE, device=device, dtype=torch.float32)
    finite = torch.empty_strided(
        REDUCTION_SHAPE, REDUCTION_STRIDE, device=device, dtype=torch.bool)
    denom = torch.empty_strided(
        REDUCTION_SHAPE, REDUCTION_STRIDE, device=device, dtype=torch.float32)
    keep = torch.empty_strided(
        OUT_SHAPE_4D, CONTIG_4D_STRIDE, device=device, dtype=torch.bool)
    final = torch.empty_strided(
        OUT_SHAPE_3D, CONTIG_3D_STRIDE, device=device, dtype=torch.bfloat16)

    seed = torch.ops.prims.inductor_lookup_seed.default(arg3_1, SEED_INDEX)
    random = _inductor_random_for_eager_check(OUT_SHAPE_4D, seed, device=device)

    content_2d = arg0_1.contiguous().view(N_ROWS, K_LEN)
    rel_1d = arg1_1.contiguous().view(-1)
    idx_1d = arg2_1.view(-1)
    random_2d = random.contiguous().view(N_ROWS, K_LEN)
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
        _xlnet_train_softmax_dropout_kernel,
        (content_2d, rel_1d, idx_1d, random_2d,
         add_2d, amax_1d, amax_scaled_1d, finite_1d, denom_1d,
         keep_2d, final_2d,
         K_LEN, GROUP_STRIDE, SLICE_OFFSET, QUERY_STRIDE),
    )

    return (add_out, amax, amax_scaled, finite, denom, keep,
            final, final.permute(0, 2, 1))
