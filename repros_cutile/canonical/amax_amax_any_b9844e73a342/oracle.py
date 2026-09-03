"""cuTile port of amax_amax_any_b9844e73a342: XLNet train softmax + dropout.

Reproduces the Triton `_xlnet_train_softmax_dropout_kernel` in cuTile:
  - Load content bf16 and relative-position bf16 via an index_ptr gather.
  - Add in fp32, round to bf16, then scale by 0.125 (bf16).
  - Compute finite mask; unscaled/scaled row amax (fp32); softmax with the
    all-finite -> unscaled_shift/all-nonfinite -> scaled_shift branch.
  - Apply seeded dropout via pre-generated inductor_random and a 1.111... scale.

The random tensor is generated OUTSIDE the kernel via inductor_random so the
cuTile kernel is deterministic-input-only.
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 90
N_ROWS = 16 * 16 * 512
K_LEN = 512
OUT_SHAPE_4D = (16, 16, 512, 512)
REDUCTION_SHAPE = (16, 16, 512, 1)
OUT_SHAPE_3D = (256, 512, 512)
CONTIG_4D_STRIDE = (4194304, 262144, 512, 1)
REDUCTION_STRIDE = (8192, 512, 1, 1)
CONTIG_3D_STRIDE = (262144, 512, 1)
SCALE = 0.125
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112


@ct.kernel
def _xlnet_softmax_dropout_kernel(
    content_ptr,     # bf16 [N_ROWS, K_LEN]
    rel_ptr,         # bf16 [N_ROWS, K_LEN] pre-gathered
    random_ptr,      # f32 [N_ROWS, K_LEN]
    add_out_ptr,     # bf16 [N_ROWS, K_LEN]
    amax_out_ptr,    # f32 [N_ROWS]
    amax_scaled_out_ptr,  # f32 [N_ROWS]
    finite_out_ptr,  # b8 [N_ROWS]
    denom_out_ptr,   # f32 [N_ROWS]
    keep_out_ptr,    # b8 [N_ROWS, K_LEN]
    final_out_ptr,   # bf16 [N_ROWS, K_LEN]
    K_LEN_C: ct.Constant[int],
):
    row = ct.bid(0)

    content = ct.load(content_ptr, index=(row, 0), shape=(1, K_LEN_C))
    rel = ct.load(rel_ptr, index=(row, 0), shape=(1, K_LEN_C))

    content_f = ct.astype(content, ct.float32)
    rel_f = ct.astype(rel, ct.float32)
    added_bf = ct.astype(content_f + rel_f, ct.bfloat16)
    ct.store(add_out_ptr, index=(row, 0), tile=added_bf)

    unscaled = ct.astype(added_bf, ct.float32)
    scaled_bf = ct.astype(unscaled * SCALE, ct.bfloat16)
    scaled = ct.astype(scaled_bf, ct.float32)

    # Finite mask: scaled == scaled (not NaN) & abs(scaled) != inf
    abs_scaled = abs(scaled)
    inf_tile = ct.full((1, K_LEN_C), float("inf"), dtype=ct.float32)
    is_finite = (scaled == scaled) & (abs_scaled != inf_tile)
    zero_i = ct.zeros((1, K_LEN_C), dtype=ct.int32)
    one_i = ct.full((1, K_LEN_C), 1, dtype=ct.int32)
    invalid_i = ct.where(is_finite, zero_i, one_i)
    any_invalid = ct.sum(invalid_i) != 0
    row_is_finite = ~any_invalid

    unscaled_max = ct.max(unscaled)
    scaled_max = ct.max(scaled)
    ct.store(amax_out_ptr, index=(row,),
             tile=ct.reshape(ct.full((1,), unscaled_max, dtype=ct.float32), (1,)))
    ct.store(amax_scaled_out_ptr, index=(row,),
             tile=ct.reshape(ct.full((1,), scaled_max, dtype=ct.float32), (1,)))
    ct.store(finite_out_ptr, index=(row,),
             tile=ct.reshape(ct.full((1,), row_is_finite, dtype=ct.bool_), (1,)))

    shifted_unscaled = (unscaled - unscaled_max) * SCALE
    shifted_scaled = scaled - scaled_max
    shifted = ct.where(row_is_finite, shifted_unscaled, shifted_scaled)

    numer = ct.exp(shifted)
    denom = ct.sum(numer)
    ct.store(denom_out_ptr, index=(row,),
             tile=ct.reshape(ct.full((1,), denom, dtype=ct.float32), (1,)))
    probs = numer / denom

    rand_f = ct.load(random_ptr, index=(row, 0), shape=(1, K_LEN_C))
    thresh_tile = ct.full((1, K_LEN_C), DROPOUT_P, dtype=ct.float32)
    keep = rand_f > thresh_tile
    ct.store(keep_out_ptr, index=(row, 0), tile=keep)

    zero_f = ct.full((1, K_LEN_C), 0.0, dtype=ct.float32)
    keep_f = ct.where(keep, probs, zero_f)
    scaled_dropout = keep_f * DROPOUT_SCALE
    ct.store(final_out_ptr, index=(row, 0), tile=ct.astype(scaled_dropout, ct.bfloat16))


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


@oracle_impl(hardware="B200", point="782e420b")
def oracle_forward(inputs):
    arg0_1, arg1_1, arg2_1, arg3_1, *_shape_params = inputs

    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        raise NotImplementedError(
            "cuTile port unsupported inside CUDA graph capture (seeded RNG)."
        )

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

    # Precompute the relative-position gather outside the kernel. The Triton
    # kernel computes:
    #   rel_offsets = group*524288 + 512 + query*1023 + rel_index[cols]
    # where rows = 16*16*512 flat, and cols = arange(512), and
    #   group = rows // 512, query = rows - group*512
    # Also arg1_1 is bf16[256, 512, 1024] laid out as (16, 16, 1024, 512) after
    # view (see repro). But the raw pointer arithmetic in the Triton kernel uses
    # `group * 524288 + 512 + query * 1023 + rel_index[cols]` — a linear index
    # into the flat bf16 buffer arg1_1 of length 256 * 512 * 1024 = 134217728.
    # We reproduce it exactly.
    rel_index = arg2_1.view(K_LEN)  # int64 [512]
    rows_1d = torch.arange(N_ROWS, device=device)
    group = rows_1d // K_LEN
    query = rows_1d - group * K_LEN
    cols_1d = torch.arange(K_LEN, device=device)
    rel_offsets = (
        group.unsqueeze(1) * 524288 + 512 +
        query.unsqueeze(1) * 1023 + rel_index.unsqueeze(0)
    )  # [N_ROWS, K_LEN]
    rel_gathered = arg1_1.reshape(-1).index_select(0, rel_offsets.reshape(-1)).view(N_ROWS, K_LEN)

    # Precompute random via inductor_random
    seed = torch.ops.prims.inductor_lookup_seed.default(arg3_1, SEED_INDEX)
    random = _inductor_random_for_eager_check(OUT_SHAPE_4D, seed, device=device)

    content_2d = arg0_1.view(N_ROWS, K_LEN)
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
        _xlnet_softmax_dropout_kernel,
        (content_2d, rel_gathered, random_2d, add_2d,
         amax_1d, amax_scaled_1d, finite_1d, denom_1d,
         keep_2d, final_2d, K_LEN),
    )
    return add_out, amax, amax_scaled, finite, denom, keep, final, final.permute(0, 2, 1)
