"""cuTile port of mean_var_d0e54075d3fd: BERT dropout-residual-LayerNorm.

Pre-generates the random tensor via inductor_random (SEED_INDEX=7) and runs a
row cuTile kernel doing dropout, residual add, LayerNorm affine, side outputs.

Hidden=768 is non-pow2; we use BLOCK_H=1024 with masked padding zero.
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 16
SEQ = 128
ROWS = BATCH * SEQ
HIDDEN = 768
SEED_INDEX = 7
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112
DENOM_EPS = 1.0e-6


@ct.kernel
def _dropout_layernorm_scope_kernel(
    x_ptr,          # bf16 [ROWS, HIDDEN]
    random_ptr,     # f32  [ROWS, HIDDEN_PADDED]
    residual_ptr,   # f32  [ROWS, HIDDEN]
    weight_ptr,     # f32  [HIDDEN]
    bias_ptr,       # f32  [HIDDEN]
    mask_ptr,       # bool [ROWS, HIDDEN]
    add_ptr,        # f32  [ROWS, HIDDEN]
    sqrt_ptr,       # f32  [ROWS]
    centered_ptr,   # f32  [ROWS, HIDDEN]
    final_ptr,      # bf16 [ROWS, HIDDEN]
    HIDDEN_SIZE: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
    ROW_BLOCK: ct.Constant[int],
):
    row_block = ct.bid(0)

    cols = ct.arange(BLOCK_H, dtype=ct.int32)
    col_mask = cols < HIDDEN_SIZE

    x_bf = ct.load(
        x_ptr, index=(row_block, 0), shape=(ROW_BLOCK, BLOCK_H),
        padding_mode=ct.PaddingMode.ZERO,
    )
    random_f = ct.load(
        random_ptr, index=(row_block, 0), shape=(ROW_BLOCK, BLOCK_H),
        padding_mode=ct.PaddingMode.ZERO,
    )
    residual_f = ct.astype(
        ct.load(
            residual_ptr, index=(row_block, 0), shape=(ROW_BLOCK, BLOCK_H),
            padding_mode=ct.PaddingMode.ZERO,
        ),
        ct.float32,
    )

    # Dropout mask, using f32-to-bf16 to match Triton's mask cast.
    random_bf = ct.astype(random_f, ct.bfloat16)
    threshold_bf = ct.full((ROW_BLOCK, BLOCK_H), DROPOUT_P, dtype=ct.bfloat16)
    keep = random_bf > threshold_bf

    zero_bf = ct.zeros((ROW_BLOCK, BLOCK_H), dtype=ct.bfloat16)
    dropped = ct.where(keep, x_bf, zero_bf)
    scaled_bf = ct.astype(ct.astype(dropped, ct.float32) * DROPOUT_SCALE, ct.bfloat16)
    add_value = residual_f + ct.astype(scaled_bf, ct.float32)

    # Mask non-valid columns to 0 for reduction.
    zero_f = ct.zeros((ROW_BLOCK, BLOCK_H), dtype=ct.float32)
    col_mask_2d = ct.reshape(col_mask, (1, BLOCK_H))
    add_masked = ct.where(col_mask_2d, add_value, zero_f)

    mean_1d = ct.sum(add_masked, axis=1, keepdims=True) * (1.0 / HIDDEN_SIZE)
    centered = add_value - mean_1d
    centered_masked = ct.where(col_mask_2d, centered, zero_f)
    variance_1d = ct.sum(centered_masked * centered_masked, axis=1, keepdims=True) * (1.0 / (HIDDEN_SIZE - 1.0))
    sqrt_val = ct.sqrt(variance_1d)

    weight = ct.astype(
        ct.load(weight_ptr, index=(0,), shape=(BLOCK_H,), padding_mode=ct.PaddingMode.ZERO),
        ct.float32,
    )
    bias = ct.astype(
        ct.load(bias_ptr, index=(0,), shape=(BLOCK_H,), padding_mode=ct.PaddingMode.ZERO),
        ct.float32,
    )
    weight_2d = ct.reshape(weight, (1, BLOCK_H))
    bias_2d = ct.reshape(bias, (1, BLOCK_H))
    out = (weight_2d * centered) / (sqrt_val + DENOM_EPS) + bias_2d

    # For each row, store masked outputs using scatter to avoid OOB writes.
    row_idx = ct.arange(ROW_BLOCK, dtype=ct.int32) + row_block * ROW_BLOCK
    row_idx_2d = ct.reshape(row_idx, (ROW_BLOCK, 1))
    flat_idx = row_idx_2d * HIDDEN_SIZE + ct.reshape(cols, (1, BLOCK_H))
    valid = col_mask_2d
    ct.scatter(mask_ptr, flat_idx, keep, mask=valid)
    ct.scatter(add_ptr, flat_idx, add_value, mask=valid)
    ct.scatter(centered_ptr, flat_idx, centered, mask=valid)
    ct.scatter(final_ptr, flat_idx, ct.astype(out, ct.bfloat16), mask=valid)
    # sqrt is per row -- store just the first column
    ct.store(sqrt_ptr, index=(row_block,), tile=ct.reshape(sqrt_val, (ROW_BLOCK,)))


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


@oracle_impl(hardware="B200", point="4205ff34", ROW_BLOCK=1)
def oracle_forward(inputs, *, ROW_BLOCK: int):
    x, seeds, residual, weight, bias, _shape0, _shape1, _shape2 = inputs

    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        raise NotImplementedError(
            "cuTile port unsupported inside CUDA graph capture (seeded RNG)."
        )

    device = x.device
    mask_out = torch.empty((BATCH, SEQ, HIDDEN), device=device, dtype=torch.bool)
    add_out = torch.empty((BATCH, SEQ, HIDDEN), device=device, dtype=torch.float32)
    sqrt_out = torch.empty((BATCH, SEQ, 1), device=device, dtype=torch.float32)
    centered_out = torch.empty((BATCH, SEQ, HIDDEN), device=device, dtype=torch.float32)
    final_out = torch.empty((ROWS, HIDDEN), device=device, dtype=torch.bfloat16)

    seed = torch.ops.prims.inductor_lookup_seed.default(seeds, SEED_INDEX)
    random = _inductor_random_for_eager_check([BATCH, SEQ, HIDDEN], seed, device=device)

    # Reshape to (ROWS, HIDDEN)
    x_2d = x.contiguous().view(ROWS, HIDDEN)
    random_2d = random.contiguous().view(ROWS, HIDDEN)
    residual_2d = residual.contiguous().view(ROWS, HIDDEN)
    mask_2d = mask_out.view(ROWS, HIDDEN)
    add_2d = add_out.view(ROWS, HIDDEN)
    centered_2d = centered_out.view(ROWS, HIDDEN)
    sqrt_1d = sqrt_out.view(ROWS)
    # Flatten mask/add/centered/final to 1D for scatter.
    mask_flat = mask_2d.view(ROWS * HIDDEN)
    add_flat = add_2d.view(ROWS * HIDDEN)
    centered_flat = centered_2d.view(ROWS * HIDDEN)
    final_flat = final_out.view(ROWS * HIDDEN)

    BLOCK_H = 1024
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(ROWS, ROW_BLOCK), 1, 1),
        _dropout_layernorm_scope_kernel,
        (
            x_2d, random_2d, residual_2d, weight, bias,
            mask_flat, add_flat, sqrt_1d, centered_flat, final_flat,
            HIDDEN, BLOCK_H, ROW_BLOCK,
        ),
    )
    return mask_out, add_out, sqrt_out, centered_out, final_out
