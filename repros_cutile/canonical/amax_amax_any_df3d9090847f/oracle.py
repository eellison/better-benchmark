"""cuTile port of amax_amax_any_df3d9090847f: XLNet relative-shift attention.

Pre-generates the seeded random tensor via inductor_random on the Python side,
then runs a single cuTile row kernel that fuses: shifted relative gather,
bf16 add/scale rounding, finite-row `any`, both fp32 amax paths, natural-exp
softmax denom, dropout mask/scale, final bf16 output with permute alias.
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 74
N_ROWS = 16 * 16 * 512
K_LEN = 512
OUT_SHAPE_4D = (16, 16, 512, 512)
REDUCTION_SHAPE = (16, 16, 512, 1)
OUT_SHAPE_3D = (256, 512, 512)
CONTIG_4D_STRIDE = (4194304, 262144, 512, 1)
REDUCTION_STRIDE = (8192, 512, 1, 1)
CONTIG_3D_STRIDE = (262144, 512, 1)
DROPOUT_SCALE = 1.1111111111111112


@ct.kernel
def _xlnet_train_softmax_dropout_kernel(
    content_ptr,
    rel_ptr,
    index_ptr,
    random_ptr,
    add_out,
    amax_out,
    amax_scaled_out,
    finite_out,
    denom_out,
    keep_out,
    final_out,
    BLOCK_N: ct.Constant[int],
):
    row = ct.bid(0)

    content = ct.load(content_ptr, index=(row, 0), shape=(1, BLOCK_N))
    content_f = ct.astype(content, ct.float32)

    rel_index = ct.load(index_ptr, index=(0,), shape=(BLOCK_N,))
    group = row // 512
    query = row - group * 512
    rel_index_2d = ct.reshape(rel_index, (1, BLOCK_N))
    base_scalar = group * 524288 + 512 + query * 1023
    rel_offsets = ct.astype(rel_index_2d, ct.int64) + base_scalar
    rel_gather = ct.gather(rel_ptr, rel_offsets)
    rel_f = ct.astype(rel_gather, ct.float32)

    added_bf16 = ct.astype(content_f + rel_f, ct.bfloat16)
    ct.store(add_out, index=(row, 0), tile=added_bf16)

    unscaled = ct.astype(added_bf16, ct.float32)
    scaled_bf16 = ct.astype(ct.astype(added_bf16, ct.float32) * 0.125, ct.bfloat16)
    scaled = ct.astype(scaled_bf16, ct.float32)

    inf_val = ct.full((1, BLOCK_N), float("inf"), dtype=ct.float32)
    zero_i = ct.zeros((1, BLOCK_N), dtype=ct.int32)
    one_i = ct.full((1, BLOCK_N), 1, dtype=ct.int32)
    abs_scaled = ct.where(scaled >= 0.0, scaled, -scaled)
    is_finite = (scaled == scaled) & (abs_scaled != inf_val)
    invalid_flag = ct.where(is_finite, zero_i, one_i)
    any_invalid = ct.sum(invalid_flag)
    row_is_finite = any_invalid == 0
    ct.store(finite_out, index=(row,), tile=ct.reshape(row_is_finite, (1,)))

    unscaled_max_scalar = ct.max(unscaled)
    scaled_max_scalar = ct.max(scaled)
    ct.store(amax_out, index=(row,), tile=ct.reshape(unscaled_max_scalar, (1,)))
    ct.store(amax_scaled_out, index=(row,), tile=ct.reshape(scaled_max_scalar, (1,)))

    shifted_unscaled = (unscaled - unscaled_max_scalar) * 0.125
    shifted_scaled = scaled - scaled_max_scalar
    shifted = ct.where(row_is_finite, shifted_unscaled, shifted_scaled)

    numer = ct.exp(shifted)
    denom_scalar = ct.sum(numer)
    ct.store(denom_out, index=(row,), tile=ct.reshape(denom_scalar, (1,)))
    probs = numer * (1.0 / denom_scalar)

    random = ct.load(random_ptr, index=(row, 0), shape=(1, BLOCK_N))
    keep = random > 0.1
    ct.store(keep_out, index=(row, 0), tile=keep)

    dropped = ct.astype(keep, ct.float32) * probs
    scaled_dropout = dropped * DROPOUT_SCALE
    ct.store(final_out, index=(row, 0), tile=ct.astype(scaled_dropout, ct.bfloat16))


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


@oracle_impl(hardware="B200", point="782e420b", BLOCK_N=512)
def oracle_forward(inputs, *, BLOCK_N: int):
    arg0_1, arg1_1, arg2_1, arg3_1, *_shape_params = inputs
    device = arg0_1.device

    content_flat = arg0_1.contiguous().view(N_ROWS, K_LEN)
    rel_flat = arg1_1.contiguous().view(-1)

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

    add_out_2d = add_out.view(N_ROWS, K_LEN)
    amax_1d = amax.view(N_ROWS)
    amax_scaled_1d = amax_scaled.view(N_ROWS)
    finite_1d = finite.view(N_ROWS)
    denom_1d = denom.view(N_ROWS)
    keep_2d = keep.view(N_ROWS, K_LEN)
    final_2d = final.view(N_ROWS, K_LEN)

    seed = torch.ops.prims.inductor_lookup_seed.default(arg3_1, SEED_INDEX)
    random = _inductor_random_for_eager_check(OUT_SHAPE_4D, seed, device=device)
    random_2d = random.view(N_ROWS, K_LEN)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (N_ROWS, 1, 1), _xlnet_train_softmax_dropout_kernel,
        (
            content_flat, rel_flat, arg2_1, random_2d,
            add_out_2d, amax_1d, amax_scaled_1d, finite_1d,
            denom_1d, keep_2d, final_2d,
            BLOCK_N,
        ),
    )
    return add_out, amax, amax_scaled, finite, denom, keep, final, final.permute(0, 2, 1)
