"""cuTile port of sum_sum_sum_bb06617cadbd: Inception avg-pool-backward + add
fanout + ReLU-gated BN backward inputs.

Ports the Triton `_add2_kernel` (3x3 avg-pool backward on a channels-last bf16
tensor, followed by three bf16 add stages) and `_branch_grad_from_add2_kernel`
(compute pre-BN affine, ReLU-mask, produce branch grad-input) to cuTile.
The channel reductions (grad_bias, grad_weight, grad_input) are then computed
via `torch.ops.aten.native_batch_norm_backward`, matching the Triton oracle.

The bf16 tensors are channels-last strided; each row of the [N*H*W, C] flat
view is one (n, h, w) with C channels contiguous, so cuTile can index them
via a linear row id. cuTile requires power-of-two load tile widths, so we
pad C=768 to 1024, C_BR=96 to 128, and C_BR=384 to 512 with masked loads.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


N = 128
H = 17
W = 17
C = 768
BLOCK_C = 1024
POOL_SCALE = 0.1111111111111111


@ct.kernel
def _pool_add_stages_kernel(
    pool_ptr,      # bf16 [N*H*W, C]  channels-last flat view
    add0_ptr,      # bf16 [N*H*W, C]
    add1_ptr,      # bf16 [N*H*W, C]
    add2_ptr,      # bf16 [N*H*W, C]
    out_ptr,       # bf16 [N*H*W, C]
    H_: ct.Constant[int],
    W_: ct.Constant[int],
    C_: ct.Constant[int],
    BLOCK_C_: ct.Constant[int],
):
    n = ct.bid(0)
    h = ct.bid(1)
    w = ct.bid(2)
    row = n * (H_ * W_) + h * W_ + w

    up1 = h > 0
    up2 = h > 1
    left1 = w > 0
    left2 = w > 1

    col_idx = ct.arange(BLOCK_C_, dtype=ct.int32)
    col_mask_1d = col_idx < C_
    col_mask = ct.reshape(col_mask_1d, (1, BLOCK_C_))

    def load_pool(dh, dw, valid):
        nb = row + dh * W_ + dw
        val_bf = ct.load(
            pool_ptr, index=(nb, 0), shape=(1, BLOCK_C_),
            padding_mode=ct.PaddingMode.ZERO,
        )
        # bf16 -> f32 * scale -> bf16 -> f32 (matches Triton roundtrip)
        scaled_f32 = ct.astype(val_bf, ct.float32) * POOL_SCALE
        scaled_bf = ct.astype(scaled_f32, ct.bfloat16)
        scaled_back = ct.astype(scaled_bf, ct.float32)
        factor = ct.astype(valid, ct.float32)
        return scaled_back * factor

    pool = load_pool(0, 0, True)
    pool = pool + load_pool(-1, 0, up1)
    pool = pool + load_pool(-2, 0, up2)
    pool = pool + load_pool(0, -1, left1)
    pool = pool + load_pool(0, -2, left2)
    pool = pool + load_pool(-1, -1, up1 & left1)
    pool = pool + load_pool(-1, -2, up1 & left2)
    pool = pool + load_pool(-2, -1, up2 & left1)
    pool = pool + load_pool(-2, -2, up2 & left2)
    pooled = ct.astype(ct.astype(pool, ct.bfloat16), ct.float32)

    add0 = ct.load(add0_ptr, index=(row, 0), shape=(1, BLOCK_C_),
                   padding_mode=ct.PaddingMode.ZERO)
    stage0 = ct.astype(ct.astype(pooled + ct.astype(add0, ct.float32), ct.bfloat16), ct.float32)
    add1 = ct.load(add1_ptr, index=(row, 0), shape=(1, BLOCK_C_),
                   padding_mode=ct.PaddingMode.ZERO)
    stage1 = ct.astype(ct.astype(stage0 + ct.astype(add1, ct.float32), ct.bfloat16), ct.float32)
    add2 = ct.load(add2_ptr, index=(row, 0), shape=(1, BLOCK_C_),
                   padding_mode=ct.PaddingMode.ZERO)
    source = ct.astype(stage1 + ct.astype(add2, ct.float32), ct.bfloat16)
    # Only store the first C columns via masked store — pack into a padded
    # buffer and copy back below.
    ct.store(out_ptr, index=(row, 0), tile=source)


@ct.kernel
def _branch_grad_kernel(
    source_ptr,     # bf16 [N*H*W, BLOCK_BR]  add2 slice at source-offset column range
    x_ptr,          # bf16 [N*H*W, BLOCK_BR] branch input (channels-last flat view, padded)
    mean_ptr,       # f32 [BLOCK_BR]
    invstd_ptr,     # f32 [BLOCK_BR]
    weight_ptr,     # f32 [BLOCK_BR]
    bias_ptr,       # f32 [BLOCK_BR]
    grad_ptr,       # bf16 [N*H*W, BLOCK_BR]
    ZERO_VAL: ct.Constant[float],
    BLOCK_BR: ct.Constant[int],
):
    row = ct.bid(0)

    x_bf = ct.load(x_ptr, index=(row, 0), shape=(1, BLOCK_BR),
                   padding_mode=ct.PaddingMode.ZERO)
    x = ct.astype(x_bf, ct.float32)
    mean = ct.load(mean_ptr, index=(0,), shape=(BLOCK_BR,),
                   padding_mode=ct.PaddingMode.ZERO)
    invstd = ct.load(invstd_ptr, index=(0,), shape=(BLOCK_BR,),
                     padding_mode=ct.PaddingMode.ZERO)
    weight = ct.load(weight_ptr, index=(0,), shape=(BLOCK_BR,),
                     padding_mode=ct.PaddingMode.ZERO)
    bias = ct.load(bias_ptr, index=(0,), shape=(BLOCK_BR,),
                   padding_mode=ct.PaddingMode.ZERO)
    mean_2d = ct.reshape(mean, (1, BLOCK_BR))
    invstd_2d = ct.reshape(invstd, (1, BLOCK_BR))
    weight_2d = ct.reshape(weight, (1, BLOCK_BR))
    bias_2d = ct.reshape(bias, (1, BLOCK_BR))

    affine = (x - mean_2d) * invstd_2d * weight_2d + bias_2d
    affine_bf16 = ct.astype(affine, ct.bfloat16)
    affine_back = ct.astype(affine_bf16, ct.float32)

    # Source tensor is a slice at columns [SOURCE_OFFSET, SOURCE_OFFSET + BLOCK_BR).
    source_bf = ct.load(source_ptr, index=(row, 0), shape=(1, BLOCK_BR),
                        padding_mode=ct.PaddingMode.ZERO)
    source_f32 = ct.astype(source_bf, ct.float32)

    # ReLU gate: where affine_bf16 <= 0, use zero constant; else use source.
    zero_tile = ct.full(shape=(1, BLOCK_BR), fill_value=ZERO_VAL, dtype=ct.float32)
    keep_source = affine_back > 0.0
    grad_f32 = ct.where(keep_source, source_f32, zero_tile)
    ct.store(grad_ptr, index=(row, 0), tile=ct.astype(grad_f32, ct.bfloat16))


def _bn_backward(grad, x, mean, invstd, weight):
    grad_input, grad_weight, grad_bias = torch.ops.aten.native_batch_norm_backward.default(
        grad,
        x,
        weight,
        None,
        None,
        mean.reshape(-1),
        invstd.reshape(-1),
        True,
        0.0,
        [True, True, True],
    )
    return grad_bias, grad_weight, grad_input


def _flat_view(tensor):
    """View a channels-last [N, C, H, W] tensor as a contiguous [N*H*W, C] tile."""
    n, c, h, w = tensor.shape
    return torch.as_strided(tensor, (n * h * w, c), (c, 1))


def _flat_padded(tensor, block_c):
    """Return an [N*H*W, block_c] padded contiguous buffer initialized from tensor."""
    n, c, h, w = tensor.shape
    n_rows = n * h * w
    padded = torch.zeros((n_rows, block_c), device=tensor.device, dtype=tensor.dtype)
    padded[:, :c] = _flat_view(tensor)
    return padded


@oracle_impl(hardware="B200", point="0d731111")
def oracle_forward(inputs, **_kwargs):
    (
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1,
        arg5_1, arg6_1, arg7_1, arg8_1, arg9_1,
        arg10_1,
        arg11_1, arg12_1, arg13_1, arg14_1, arg15_1,
    ) = inputs
    del arg1_1  # unused (avg_pool2d_backward's kernel_size/etc. are constexpr)

    device = arg0_1.device
    n_rows = N * H * W

    # Allocate padded pool_flat / add_flat inputs (C=768 -> BLOCK_C=1024) so we can
    # use power-of-two loads. Only the first C columns are meaningful.
    pool_padded = _flat_padded(arg0_1, BLOCK_C)
    add0_padded = _flat_padded(arg2_1, BLOCK_C)
    add1_padded = _flat_padded(arg3_1, BLOCK_C)
    add2_padded = _flat_padded(arg4_1, BLOCK_C)
    out_padded = torch.empty((n_rows, BLOCK_C), device=device, dtype=torch.bfloat16)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (N, H, W),
        _pool_add_stages_kernel,
        (pool_padded, add0_padded, add1_padded, add2_padded, out_padded,
         H, W, C, BLOCK_C),
    )

    # Copy padded output back to channels-last strided out_add2.
    out_add2 = torch.empty_strided(
        tuple(arg0_1.shape), tuple(arg0_1.stride()),
        device=device, dtype=torch.bfloat16,
    )
    _flat_view(out_add2).copy_(out_padded[:, :C])

    # Scalar for the ReLU-clipped grad: arg10_1 is bf16[]. Match Triton's
    # `tl.load(zero_ptr).to(tl.float32)` — cast the bf16 value to f32 exactly.
    zero_val = float(arg10_1.to(torch.float32).item())

    def _pad1d(vec, block):
        buf = torch.zeros((block,), device=device, dtype=vec.dtype)
        n = vec.numel()
        buf[:n] = vec.reshape(-1)
        return buf

    def _make_source_slice(source_col_start, c_br, block_br):
        """Extract [source_col_start, source_col_start + c_br) columns of add_2
        into a padded [n_rows, block_br] contiguous buffer."""
        buf = torch.zeros((n_rows, block_br), device=device, dtype=torch.bfloat16)
        add_2_view = _flat_view(out_add2)  # [n_rows, C]
        buf[:, :c_br] = add_2_view[:, source_col_start:source_col_start + c_br]
        return buf

    # Branch 96 (channels 384..480 of add2): BLOCK_BR=128
    BLOCK_96 = 128
    x96_padded = _flat_padded(arg5_1, BLOCK_96)
    grad96_padded = torch.empty((n_rows, BLOCK_96), device=device, dtype=torch.bfloat16)
    mean96 = _pad1d(arg6_1, BLOCK_96)
    invstd96 = _pad1d(arg7_1, BLOCK_96)
    weight96 = _pad1d(arg8_1, BLOCK_96)
    bias96 = _pad1d(arg9_1, BLOCK_96)
    source96 = _make_source_slice(384, 96, BLOCK_96)

    ct.launch(
        stream,
        (n_rows, 1, 1),
        _branch_grad_kernel,
        (source96, x96_padded, mean96, invstd96, weight96, bias96, grad96_padded,
         zero_val, BLOCK_96),
    )

    grad96 = torch.empty_strided(
        tuple(arg5_1.shape), tuple(arg5_1.stride()),
        device=device, dtype=torch.bfloat16,
    )
    _flat_view(grad96).copy_(grad96_padded[:, :96])

    # Branch 384: BLOCK_BR=512
    BLOCK_384 = 512
    x384_padded = _flat_padded(arg11_1, BLOCK_384)
    grad384_padded = torch.empty((n_rows, BLOCK_384), device=device, dtype=torch.bfloat16)
    mean384 = _pad1d(arg12_1, BLOCK_384)
    invstd384 = _pad1d(arg13_1, BLOCK_384)
    weight384 = _pad1d(arg14_1, BLOCK_384)
    bias384 = _pad1d(arg15_1, BLOCK_384)
    source384 = _make_source_slice(0, 384, BLOCK_384)

    ct.launch(
        stream,
        (n_rows, 1, 1),
        _branch_grad_kernel,
        (source384, x384_padded, mean384, invstd384, weight384, bias384, grad384_padded,
         zero_val, BLOCK_384),
    )

    grad384 = torch.empty_strided(
        tuple(arg11_1.shape), tuple(arg11_1.stride()),
        device=device, dtype=torch.bfloat16,
    )
    _flat_view(grad384).copy_(grad384_padded[:, :384])

    out96 = _bn_backward(grad96, arg5_1, arg6_1, arg7_1, arg8_1)
    out384 = _bn_backward(grad384, arg11_1, arg12_1, arg13_1, arg14_1)
    return (out_add2,) + out96 + out384
