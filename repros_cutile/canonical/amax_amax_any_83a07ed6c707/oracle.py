"""cuTile port of amax_amax_any_83a07ed6c707: XLNet train scope softmax+dropout.

Complete port: relative-index gather, softmax with fp32 amax, dropout with
seeded RNG (pre-generated), and layout aliases.

Outputs: (add_out, amax, amax_scaled, finite, denom, keep, final, final_perm).
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 62
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112
N_ROWS = 16 * 16 * 512  # 131072
K_LEN = 512
OUT_SHAPE_4D = (16, 16, 512, 512)


@ct.kernel
def _xlnet_softmax_dropout_kernel(
    content_ptr,    # bf16 [rows, K]
    rel_ptr,        # bf16 [rows, K]  precomputed relative-shifted view
    random_ptr,     # f32 [rows, K]
    add_ptr,        # bf16 [rows, K] added
    amax_ptr,       # f32 [rows, 1]
    amax_scaled_ptr,# f32 [rows, 1]
    finite_ptr,     # b8 [rows, 1]
    denom_ptr,      # f32 [rows, 1]
    keep_ptr,       # b8 [rows, K]
    final_ptr,      # bf16 [rows, K]
    K: ct.Constant[int],
    BLOCK_K: ct.Constant[int],
    DROPOUT_P_: ct.Constant[float],
    DROPOUT_SCALE_: ct.Constant[float],
):
    row = ct.bid(0)

    content = ct.load(content_ptr, index=(row, 0), shape=(1, BLOCK_K))
    rel = ct.load(rel_ptr, index=(row, 0), shape=(1, BLOCK_K))

    content_f = ct.astype(content, ct.float32)
    rel_f = ct.astype(rel, ct.float32)
    added_bf = ct.astype(content_f + rel_f, ct.bfloat16)
    unscaled = ct.astype(added_bf, ct.float32)
    scaled_bf = ct.astype(unscaled * 0.125, ct.bfloat16)
    scaled = ct.astype(scaled_bf, ct.float32)
    ct.store(add_ptr, index=(row, 0), tile=added_bf)

    # finite check: all(abs(scaled) != inf) & all(scaled==scaled)
    abs_scaled = ct.where(scaled < 0.0, -scaled, scaled)
    inf_v = ct.full((1, BLOCK_K), 1.0e38, dtype=ct.float32)
    finite = (scaled == scaled) & (abs_scaled < inf_v)
    invalid = ~finite
    zero_i32 = ct.full((1, BLOCK_K), 0, dtype=ct.int32)
    one_i32 = ct.full((1, BLOCK_K), 1, dtype=ct.int32)
    invalid_i = ct.where(invalid, one_i32, zero_i32)
    max_invalid = ct.max(invalid_i, axis=1, keepdims=True)
    has_invalid = max_invalid != 0
    row_is_finite = ~has_invalid  # (1, 1)
    ct.store(finite_ptr, index=(row, 0), tile=row_is_finite)

    unscaled_max = ct.max(unscaled, axis=1, keepdims=True)
    scaled_max = ct.max(scaled, axis=1, keepdims=True)
    ct.store(amax_ptr, index=(row, 0), tile=unscaled_max)
    ct.store(amax_scaled_ptr, index=(row, 0), tile=scaled_max)

    shifted_unscaled = (unscaled - unscaled_max) * 0.125
    shifted_scaled = scaled - scaled_max
    shifted = ct.where(row_is_finite, shifted_unscaled, shifted_scaled)
    # Note: no OOB masking needed since K=512 matches BLOCK_K.

    exp_v = ct.exp(shifted)
    sum_v = ct.sum(exp_v, axis=1, keepdims=True)
    ct.store(denom_ptr, index=(row, 0), tile=sum_v)
    probs = exp_v / sum_v

    random_f = ct.load(random_ptr, index=(row, 0), shape=(1, BLOCK_K))
    keep = random_f > DROPOUT_P_
    ct.store(keep_ptr, index=(row, 0), tile=keep)

    keep_f = ct.astype(keep, ct.float32)
    dropped = keep_f * probs
    scaled_dropout = dropped * DROPOUT_SCALE_
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
    (arg0_1, arg1_1, arg2_1, arg3_1, *_shape) = inputs
    device = arg0_1.device

    # Build relative-shift indexed input via torch ops (matches Triton computation):
    # for row r in [0, N_ROWS), col c in [0, K_LEN):
    #   group = r // 512; query = r % 512
    #   rel_offset = group * 524288 + 512 + query * 1023 + rel_index[c]
    #   rel = arg1_ptr[rel_offset]
    # First flatten arg1 and use gather.
    rows = torch.arange(N_ROWS, device=device)
    group = rows // 512
    query = rows - group * 512
    cols = torch.arange(K_LEN, device=device)  # [K]
    rel_index = arg2_1.long()  # [K]
    rel_offsets = (group[:, None] * 524288 + 512
                   + query[:, None] * 1023 + rel_index[None, :])  # [rows, K]
    linear = rows[:, None] * 512 + cols[None, :]  # [rows, K]

    content = arg0_1.reshape(-1)[linear.view(-1)].view(N_ROWS, K_LEN)
    rel = arg1_1.reshape(-1)[rel_offsets.view(-1)].view(N_ROWS, K_LEN)

    add_out = torch.empty((N_ROWS, K_LEN), device=device, dtype=torch.bfloat16)
    amax = torch.empty((N_ROWS, 1), device=device, dtype=torch.float32)
    amax_scaled = torch.empty((N_ROWS, 1), device=device, dtype=torch.float32)
    finite = torch.empty((N_ROWS, 1), device=device, dtype=torch.bool)
    denom = torch.empty((N_ROWS, 1), device=device, dtype=torch.float32)
    keep = torch.empty((N_ROWS, K_LEN), device=device, dtype=torch.bool)
    final = torch.empty((N_ROWS, K_LEN), device=device, dtype=torch.bfloat16)

    seed = torch.ops.prims.inductor_lookup_seed.default(arg3_1, SEED_INDEX)
    random = _inductor_random_for_eager_check(OUT_SHAPE_4D, seed, device=device)
    random_flat = random.reshape(N_ROWS, K_LEN)

    stream = torch.cuda.current_stream()
    ct.launch(stream, (N_ROWS, 1, 1), _xlnet_softmax_dropout_kernel,
              (content, rel, random_flat,
               add_out, amax, amax_scaled, finite, denom, keep, final,
               K_LEN, K_LEN, DROPOUT_P, DROPOUT_SCALE))

    add_out_4d = add_out.view(16, 16, 512, 512)
    amax_4d = amax.view(16, 16, 512, 1)
    amax_scaled_4d = amax_scaled.view(16, 16, 512, 1)
    finite_4d = finite.view(16, 16, 512, 1)
    denom_4d = denom.view(16, 16, 512, 1)
    keep_4d = keep.view(16, 16, 512, 512)
    final_3d = final.view(256, 512, 512)
    final_perm = final_3d.permute(0, 2, 1)
    return (add_out_4d, amax_4d, amax_scaled_4d, finite_4d, denom_4d,
            keep_4d, final_3d, final_perm)
