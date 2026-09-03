"""cuTile port of var_mean_62d1a04fba6c: ConvBERT dropout + residual LayerNorm.

Pre-generates the seeded random tensor via inductor_random on the Python side,
then runs a single cuTile row kernel that: applies dropout mask, adds
residual (f32), performs LayerNorm (eps=1e-12), affine, and emits multiple
outputs (gt mask, normalized f32, affine f32, bf16 view, bf16 permuted, invstd/H).
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 16
DROPOUT_SCALE = 1.1111111111111112
EPS = 1.0e-12


@ct.kernel
def _dropout_residual_ln_kernel(
    addmm_ptr,       # bf16 (rows, HIDDEN)
    random_ptr,      # f32 (rows, HIDDEN) — pre-generated
    residual_ptr,    # f32 (rows, HIDDEN)
    weight_ptr,      # f32 (HIDDEN,)
    bias_ptr,        # f32 (HIDDEN,)
    gt_ptr,          # bool (rows, HIDDEN)
    normalized_ptr,  # f32 (rows, HIDDEN)
    affine_ptr,      # f32 (rows, HIDDEN)
    bf16_view_ptr,   # bf16 (rows, HIDDEN)
    bf16_perm_ptr,   # bf16 (rows, HIDDEN) — permuted layout
    div_ptr,         # f32 (rows,)
    HIDDEN: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
):
    row = ct.bid(0)

    addmm = ct.load(addmm_ptr, index=(row, 0), shape=(1, BLOCK_H),
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
    dropped_bf = ct.where(keep, addmm, zero_bf)
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
    ct.scatter(bf16_view_ptr, ct.reshape(row * HIDDEN + cols, (1, BLOCK_H)),
               affine_bf, mask=col_mask_2d)
    ct.scatter(bf16_perm_ptr, ct.reshape(row * HIDDEN + cols, (1, BLOCK_H)),
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


def _inductor_random_for_eager_check(shape, seed, *, device):
    """Generate the same inductor_random tensor the Repro would.

    Uses the CUDA RNG state's offset to rewind for CUDA-graph-capture-parity.
    """
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


@oracle_impl(hardware="B200", point="d429ff7b", BLOCK_H=1024)
def oracle_forward(inputs, *, BLOCK_H: int):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, shape0, _shape1, shape2 = inputs
    norm_shape = _as_shape(shape0)
    flat_shape = _as_shape(shape2)
    rows = int(arg0_1.shape[0])
    hidden = int(arg3_1.shape[0])
    div_shape = (norm_shape[0], norm_shape[1], 1)
    permuted_shape = (norm_shape[0], hidden, norm_shape[1])
    permuted_stride = (norm_shape[1] * hidden, 1, hidden)

    gt = torch.empty_strided(norm_shape, _contiguous_stride(norm_shape),
                             device=arg0_1.device, dtype=torch.bool)
    normalized = torch.empty_strided(norm_shape, _contiguous_stride(norm_shape),
                                     device=arg0_1.device, dtype=torch.float32)
    affine = torch.empty_strided(norm_shape, _contiguous_stride(norm_shape),
                                 device=arg0_1.device, dtype=torch.float32)
    bf16_view = torch.empty_strided(flat_shape, _contiguous_stride(flat_shape),
                                    device=arg0_1.device, dtype=torch.bfloat16)
    bf16_permute = torch.empty_strided(permuted_shape, permuted_stride,
                                       device=arg0_1.device, dtype=torch.bfloat16)
    div = torch.empty_strided(div_shape, _contiguous_stride(div_shape),
                              device=arg0_1.device, dtype=torch.float32)

    seed = torch.ops.prims.inductor_lookup_seed.default(arg1_1, SEED_INDEX)
    random = _inductor_random_for_eager_check(norm_shape, seed, device=arg0_1.device)
    random_2d = random.view(rows, hidden)
    resid_2d = arg2_1.view(rows, hidden)
    normalized_2d = normalized.view(rows, hidden)
    affine_2d = affine.view(rows, hidden)
    bf16_view_2d = bf16_view.view(rows, hidden)
    gt_2d = gt.view(rows, hidden)
    div_1d = div.view(rows)

    total_elems = rows * hidden
    # bf16_permute has strides (norm_shape[1]*hidden, 1, hidden); its storage is
    # (norm_shape[0], hidden, norm_shape[1]) contiguous. View that same storage
    # as a flat 1D array for the scatter into permuted layout.
    bf16_perm_flat = bf16_permute.as_strided((total_elems,), (1,),
                                             storage_offset=bf16_permute.storage_offset())

    gt_flat = gt_2d.reshape(total_elems)
    normalized_flat = normalized_2d.reshape(total_elems)
    affine_flat = affine_2d.reshape(total_elems)
    bf16_view_flat = bf16_view_2d.reshape(total_elems)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (rows, 1, 1),
        _dropout_residual_ln_kernel,
        (
            arg0_1, random_2d, resid_2d, arg3_1, arg4_1,
            gt_flat, normalized_flat, affine_flat, bf16_view_flat, bf16_perm_flat, div_1d,
            hidden, BLOCK_H,
        ),
    )
    return gt, normalized, affine, bf16_view, bf16_permute, div
