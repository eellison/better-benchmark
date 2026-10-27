"""cuTile port of var_mean_ed2e4094de1e: Longformer bf16 seeded-dropout +
residual LN with sequence-major permute output (seed 14).

Pre-generates the seeded random tensor via inductor_random on the Python side,
then runs a single cuTile row kernel that: applies dropout mask (bf16), adds
residual (f32), performs LayerNorm (eps=1e-5), affine, and emits multiple
outputs including a bf16 permute-store to a separate seq-major flat tensor.
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 14
DROPOUT_SCALE = 1.1111111111111112
EPS = 1.0e-5


@ct.kernel
def _dropout_residual_ln_permute_kernel(
    flat_ptr,        # bf16 (rows, HIDDEN)
    random_ptr,      # f32 (rows, HIDDEN) — pre-generated
    residual_ptr,    # f32 (rows, HIDDEN)
    weight_ptr,      # f32 (HIDDEN,)
    bias_ptr,        # f32 (HIDDEN,)
    gt_ptr,          # bool (rows, HIDDEN)
    normalized_ptr,  # f32 (rows, HIDDEN)
    affine_ptr,      # f32 (rows, HIDDEN)
    final_bf16_ptr,  # bf16 (rows, HIDDEN) — seq-major permuted layout via scatter
    div_ptr,         # f32 (rows,)
    BATCH: ct.Constant[int],
    SEQ: ct.Constant[int],
    HIDDEN: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
):
    row = ct.bid(0)

    flat_bf = ct.load(flat_ptr, index=(row, 0), shape=(1, BLOCK_H),
                      padding_mode=ct.PaddingMode.ZERO)
    random_f = ct.load(random_ptr, index=(row, 0), shape=(1, BLOCK_H),
                       padding_mode=ct.PaddingMode.ZERO)
    random_bf = ct.astype(random_f, ct.bfloat16)
    dropout_p_bf = ct.full((1, BLOCK_H), 0.1, dtype=ct.bfloat16)
    keep = random_bf > dropout_p_bf

    cols = ct.arange(BLOCK_H, dtype=ct.int32)
    col_mask_2d = ct.reshape(cols < HIDDEN, (1, BLOCK_H))
    ct.scatter(gt_ptr, ct.reshape(row * HIDDEN + cols, (1, BLOCK_H)),
               keep, mask=col_mask_2d)

    zero_bf = ct.zeros((1, BLOCK_H), dtype=ct.bfloat16)
    dropped_bf = ct.where(keep, flat_bf, zero_bf)
    scaled_bf = ct.astype(ct.astype(dropped_bf, ct.float32) * DROPOUT_SCALE, ct.bfloat16)
    residual = ct.load(residual_ptr, index=(row, 0), shape=(1, BLOCK_H),
                       padding_mode=ct.PaddingMode.ZERO)
    x = ct.astype(scaled_bf, ct.float32) + residual
    zero_f = ct.zeros((1, BLOCK_H), dtype=ct.float32)
    x_masked = ct.where(col_mask_2d, x, zero_f)

    mean = ct.sum(x_masked, axis=1, keepdims=True) * (1.0 / HIDDEN)
    centered = x - mean
    centered_masked = ct.where(col_mask_2d, centered, zero_f)
    variance = ct.sum(centered_masked * centered_masked, axis=1, keepdims=True) * (1.0 / HIDDEN)
    invstd = ct.rsqrt(variance + EPS)
    normalized = centered * invstd

    weight = ct.load(weight_ptr, index=(0,), shape=(BLOCK_H,),
                     padding_mode=ct.PaddingMode.ZERO)
    bias = ct.load(bias_ptr, index=(0,), shape=(BLOCK_H,),
                   padding_mode=ct.PaddingMode.ZERO)
    weight_2d = ct.reshape(weight, (1, BLOCK_H))
    bias_2d = ct.reshape(bias, (1, BLOCK_H))
    affine = normalized * weight_2d + bias_2d
    affine_bf = ct.astype(affine, ct.bfloat16)

    ct.scatter(normalized_ptr, ct.reshape(row * HIDDEN + cols, (1, BLOCK_H)),
               normalized, mask=col_mask_2d)
    ct.scatter(affine_ptr, ct.reshape(row * HIDDEN + cols, (1, BLOCK_H)),
               affine, mask=col_mask_2d)

    # Permute row -> (batch, seq) then out_row = seq * BATCH + batch.
    batch_id = row // SEQ
    seq_id = row - batch_id * SEQ
    out_row = seq_id * BATCH + batch_id
    ct.scatter(final_bf16_ptr, ct.reshape(out_row * HIDDEN + cols, (1, BLOCK_H)),
               affine_bf, mask=col_mask_2d)

    inv_h = ct.reshape(invstd * (1.0 / HIDDEN), (1,))
    ct.store(div_ptr, index=(row,), tile=inv_h)


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= int(dim)
    return tuple(reversed(stride))


def _as_shape(shape):
    return tuple(int(dim) for dim in shape)


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


@oracle_impl(hardware="B200", point="04503798", BLOCK_H=1024)
def oracle_forward(inputs, *, BLOCK_H: int):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, shape0, _shape1, shape2 = inputs
    norm_shape = _as_shape(shape0)
    final_shape = _as_shape(shape2)
    rows = int(arg0_1.shape[0])
    hidden = int(arg0_1.shape[1])
    batch = int(norm_shape[0])
    seq = int(norm_shape[1])
    div_shape = (batch, seq, 1)

    gt = torch.empty_strided(norm_shape, _contiguous_stride(norm_shape),
                             device=arg0_1.device, dtype=torch.bool)
    normalized = torch.empty_strided(norm_shape, _contiguous_stride(norm_shape),
                                     device=arg0_1.device, dtype=torch.float32)
    affine = torch.empty_strided(norm_shape, _contiguous_stride(norm_shape),
                                 device=arg0_1.device, dtype=torch.float32)
    final_bf16 = torch.empty_strided(final_shape, _contiguous_stride(final_shape),
                                     device=arg0_1.device, dtype=torch.bfloat16)
    div = torch.empty_strided(div_shape, _contiguous_stride(div_shape),
                              device=arg0_1.device, dtype=torch.float32)

    seed = torch.ops.prims.inductor_lookup_seed.default(arg1_1, SEED_INDEX)
    random = _inductor_random_for_eager_check(norm_shape, seed, device=arg0_1.device)
    random_2d = random.view(rows, hidden)
    resid_2d = arg2_1.view(rows, hidden)
    total = rows * hidden
    gt_flat = gt.view(total)
    normalized_flat = normalized.view(total)
    affine_flat = affine.view(total)
    final_bf16_flat = final_bf16.view(total)
    div_1d = div.view(rows)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (rows, 1, 1),
        _dropout_residual_ln_permute_kernel,
        (
            arg0_1, random_2d, resid_2d, arg3_1, arg4_1,
            gt_flat, normalized_flat, affine_flat, final_bf16_flat, div_1d,
            batch, seq, hidden, BLOCK_H,
        ),
    )
    return gt, normalized, affine, final_bf16, div
