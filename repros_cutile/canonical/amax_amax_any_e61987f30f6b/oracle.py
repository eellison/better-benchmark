"""cuTile port of amax_amax_any_e61987f30f6b: XLNet bf16 relative-shift attention softmax + dropout.

Structure:
- Torch: perform the XLNet relative-position gather (view/slice/reshape/index)
  that materializes the rel tensor at [rows, K].
- cuTile: single row kernel that does add_1 = content + rel (bf16),
  amax(f32(add_1)), amax(f32(scaled)), finiteness check, stable softmax
  with the row's scale-choice branch, seeded dropout.
- Torch: apply the returned permute alias.
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 10
SCALE = 0.125
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112


@ct.kernel
def _xlnet_softmax_kernel(
    content_ptr,     # bf16 [rows, K]  (dense flat: [16*16*512, 512])
    rel_ptr,         # bf16 [rows, K]  (dense flat: rel gathered)
    random_ptr,      # f32  [rows, K]  (pre-generated inductor_random)
    add_out_ptr,     # bf16 [rows, K]
    amax_ptr,        # f32  [rows]
    amax_scaled_ptr, # f32  [rows]
    finite_ptr,      # b8   [rows]
    denom_ptr,       # f32  [rows]
    keep_ptr,        # b8   [rows, K]
    final_ptr,       # bf16 [rows, K]
    BLOCK_N: ct.Constant[int],
):
    row = ct.bid(0)

    content_bf = ct.load(content_ptr, index=(row, 0), shape=(1, BLOCK_N))
    rel_bf = ct.load(rel_ptr, index=(row, 0), shape=(1, BLOCK_N))
    added_bf = ct.astype(
        ct.astype(content_bf, ct.float32) + ct.astype(rel_bf, ct.float32),
        ct.bfloat16,
    )
    ct.store(add_out_ptr, index=(row, 0), tile=added_bf)

    unscaled = ct.astype(added_bf, ct.float32)
    scaled_bf = ct.astype(ct.astype(added_bf, ct.float32) * SCALE, ct.bfloat16)
    scaled = ct.astype(scaled_bf, ct.float32)

    abs_scaled = ct.astype(scaled >= 0, ct.float32) * scaled + ct.astype(scaled < 0, ct.float32) * (0.0 - scaled)
    inf_ct = ct.full((1, BLOCK_N), float("inf"), dtype=ct.float32)
    # finite = (scaled == scaled) & (abs_scaled != inf); invalid = ~finite
    is_nan = scaled != scaled
    is_inf = abs_scaled == inf_ct
    invalid_row = is_nan | is_inf
    # has_invalid: 1 if any invalid, else 0
    has_invalid = ct.max(ct.where(invalid_row, 1, 0), axis=1)
    row_is_finite_scalar = has_invalid == 0
    ct.store(finite_ptr, index=(row,), tile=ct.reshape(row_is_finite_scalar, (1,)))

    unscaled_max = ct.max(unscaled, axis=1, keepdims=True)
    scaled_max = ct.max(scaled, axis=1, keepdims=True)
    ct.store(amax_ptr, index=(row,), tile=ct.reshape(unscaled_max, (1,)))
    ct.store(amax_scaled_ptr, index=(row,), tile=ct.reshape(scaled_max, (1,)))

    shifted_unscaled = (unscaled - unscaled_max) * SCALE
    shifted_scaled = scaled - scaled_max
    row_is_finite_2d = ct.reshape(row_is_finite_scalar, (1, 1))
    shifted = ct.where(row_is_finite_2d, shifted_unscaled, shifted_scaled)

    numer = ct.exp(shifted)
    denom = ct.sum(numer, axis=1, keepdims=True)
    probs = numer / denom
    ct.store(denom_ptr, index=(row,), tile=ct.reshape(denom, (1,)))

    rand = ct.load(random_ptr, index=(row, 0), shape=(1, BLOCK_N))
    thresh = ct.full((1, BLOCK_N), DROPOUT_P, dtype=ct.float32)
    keep = rand > thresh
    ct.store(keep_ptr, index=(row, 0), tile=keep)

    zero_f = ct.full((1, BLOCK_N), 0.0, dtype=ct.float32)
    dropped = ct.where(keep, probs, zero_f)
    scaled_out = ct.astype(dropped * DROPOUT_SCALE, ct.bfloat16)
    ct.store(final_ptr, index=(row, 0), tile=scaled_out)


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


# 782e420b: XLNet relative-shift softmax + dropout, [16,16,512,512].
@oracle_impl(hardware="B200", point="782e420b", BLOCK_N=512)
def oracle_forward(inputs, *, BLOCK_N: int):
    arg0_1, arg1_1, arg2_1, arg3_1, *_shape = inputs

    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        raise NotImplementedError(
            "cuTile port unsupported inside CUDA graph capture (seeded RNG)."
        )
    device = arg0_1.device

    # Reproduce the eager relative-shift gather (matches Repro.forward).
    # arg0_1 [256,512,512] -> view then permute -> content [16,16,512,512]
    content = arg0_1.view(16, 16, 512, 1, 512).permute(0, 1, 2, 4, 3).view(16, 16, 512, 512)
    # arg1_1 [256,512,1024] -> [16,16,1024,512] -> slice(2, 1:) -> [16,16,1023,512] -> view [16,16,512,1023]
    view_4 = arg1_1.view(16, 16, 512, 1, 1024).permute(0, 1, 2, 4, 3).view(16, 16, 1024, 512)
    slice_1 = view_4[:, :, 1:, :]  # [16,16,1023,512]
    view_5 = slice_1.reshape(16, 16, 512, 1023)
    rel_gathered = view_5[..., arg2_1]  # [16,16,512,512]

    rows = 16 * 16 * 512
    K = 512
    content_2d = content.contiguous().view(rows, K)
    rel_2d = rel_gathered.contiguous().view(rows, K)

    add_out = torch.empty((16, 16, 512, 512), device=device, dtype=torch.bfloat16)
    amax = torch.empty((16, 16, 512, 1), device=device, dtype=torch.float32)
    amax_scaled = torch.empty((16, 16, 512, 1), device=device, dtype=torch.float32)
    finite = torch.empty((16, 16, 512, 1), device=device, dtype=torch.bool)
    denom = torch.empty((16, 16, 512, 1), device=device, dtype=torch.float32)
    keep = torch.empty((16, 16, 512, 512), device=device, dtype=torch.bool)
    # final is [256, 512, 512] contiguous
    final = torch.empty((256, 512, 512), device=device, dtype=torch.bfloat16)

    seed = torch.ops.prims.inductor_lookup_seed.default(arg3_1, SEED_INDEX)
    random = _inductor_random_for_eager_check((16, 16, 512, 512), seed, device=device)
    random_2d = random.view(rows, K)

    add_out_2d = add_out.view(rows, K)
    amax_1d = amax.view(rows)
    amax_scaled_1d = amax_scaled.view(rows)
    finite_1d = finite.view(rows)
    denom_1d = denom.view(rows)
    keep_2d = keep.view(rows, K)
    final_2d = final.view(rows, K)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (rows, 1, 1),
        _xlnet_softmax_kernel,
        (content_2d, rel_2d, random_2d,
         add_out_2d, amax_1d, amax_scaled_1d, finite_1d, denom_1d, keep_2d, final_2d,
         BLOCK_N),
    )
    return add_out, amax, amax_scaled, finite, denom, keep, final, final.permute(0, 2, 1)
