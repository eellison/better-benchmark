"""cuTile port of amax_amax_any_395c2caa2db4: Visformer safe scaled softmax.

Input: bf16[768, 56, 56], sliced to bf16[768, 49, 49] then viewed as
[128, 6, 49, 49]. Output includes: raw max (f32), scaled max (f32), all-finite
mask (bool), softmax denom (f32), softmax probs (bf16), and a permuted view.

Batches BLOCK_M queries per program to reduce launch overhead. K_LEN=49 is
non-pow2 so we pad columns to BLOCK_N=64. Since the input's middle dim (56)
is also non-49, we view the input as (N_HEADS*56, 56) and load rows
head*56+query — this matches Triton's row-batched kernel pattern.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


N_HEADS = 768        # = 128 * 6
K_LEN = 49           # non-power-of-2
BLOCK_M = 8          # queries per program (pow2)
BLOCK_N = 64         # next pow2 of K_LEN
SCALE = 0.08838834764831845
INPUT_W = 56         # last dim of arg0_1 (before slice)
INPUT_H = 56         # middle dim of arg0_1 (before slice)


@ct.kernel
def _visformer_safe_softmax_kernel(
    x_ptr,             # bf16 flat [N_HEADS*INPUT_H, INPUT_W]
    amax_ptr,          # f32 flat [N_HEADS*K_LEN]
    scaled_amax_ptr,   # f32 flat [N_HEADS*K_LEN]
    all_finite_ptr,    # bool flat [N_HEADS*K_LEN]
    denom_ptr,         # f32 flat [N_HEADS*K_LEN]
    out_flat_ptr,      # bf16 flat [N_HEADS*K_LEN*K_LEN]
    N_ROWS: ct.Constant[int],
    K_LEN_: ct.Constant[int],
    INPUT_H_: ct.Constant[int],
    BLOCK_M_: ct.Constant[int],
    BLOCK_N_: ct.Constant[int],
    SCALE_: ct.Constant[float],
):
    row_block = ct.bid(0)
    row_start = row_block * BLOCK_M_

    # Row indices in the compact (n_rows,) side-output/scatter layout.
    row_offs = ct.arange(BLOCK_M_, dtype=ct.int32) + row_start          # (BLOCK_M,)
    row_mask_1d = row_offs < N_ROWS
    row_mask_2d = ct.reshape(row_mask_1d, (BLOCK_M_, 1))

    # Split flat row -> (head, query) in the 49-per-head layout.
    flat_head = row_offs // K_LEN_
    query = row_offs - flat_head * K_LEN_
    # Input-space row index = head * INPUT_H + query (since input is padded 56).
    input_row = flat_head * INPUT_H_ + query                             # (BLOCK_M,)

    cols = ct.arange(BLOCK_N_, dtype=ct.int32)                           # (BLOCK_N,)
    col_mask_1d = cols < K_LEN_
    col_mask_2d = ct.reshape(col_mask_1d, (1, BLOCK_N_))

    # Gather (BLOCK_M, BLOCK_N) from the 2D-flattened input.
    input_offs = (ct.reshape(input_row, (BLOCK_M_, 1)) * ct.full((1, 1), INPUT_H_, dtype=ct.int32)
                  + ct.reshape(cols, (1, BLOCK_N_)))
    # x_ptr's second dim stride is 1; row stride is INPUT_W (= INPUT_H since square).
    # We want flat_offset = input_row * INPUT_W + col.
    # INPUT_W == INPUT_H so we reused the constant.
    valid_2d = row_mask_2d & col_mask_2d
    safe_offs = ct.where(valid_2d, input_offs, ct.zeros((BLOCK_M_, BLOCK_N_), dtype=ct.int32))
    x_bf16 = ct.gather(x_ptr, safe_offs)
    raw = ct.astype(x_bf16, ct.float32)

    neg_inf = ct.full((BLOCK_M_, BLOCK_N_), float("-inf"), dtype=ct.float32)
    raw_masked = ct.where(col_mask_2d, raw, neg_inf)

    scaled_masked = raw_masked * SCALE_

    raw_max = ct.max(raw_masked, axis=1, keepdims=True)          # (BLOCK_M, 1)
    scaled_max = ct.max(scaled_masked, axis=1, keepdims=True)    # (BLOCK_M, 1)

    abs_val = ct.where(scaled_masked >= 0.0, scaled_masked, -scaled_masked)
    inf_tile = ct.full((BLOCK_M_, BLOCK_N_), float("inf"), dtype=ct.float32)
    is_finite = (scaled_masked == scaled_masked) & (abs_val != inf_tile)
    zero_i32 = ct.zeros((BLOCK_M_, BLOCK_N_), dtype=ct.int32)
    one_i32 = ct.full((BLOCK_M_, BLOCK_N_), 1, dtype=ct.int32)
    invalid_flag = ct.where(col_mask_2d & (~is_finite), one_i32, zero_i32)
    any_invalid = ct.max(invalid_flag, axis=1, keepdims=True) != 0    # (BLOCK_M, 1)
    all_finite = ~any_invalid                                          # (BLOCK_M, 1)

    shifted_unscaled = (raw_masked - raw_max) * SCALE_
    shifted_scaled = scaled_masked - scaled_max
    shifted = ct.where(all_finite, shifted_unscaled, shifted_scaled)   # (BLOCK_M, BLOCK_N)

    numer = ct.exp(shifted)
    zero_f = ct.zeros((BLOCK_M_, BLOCK_N_), dtype=ct.float32)
    numer_masked = ct.where(col_mask_2d, numer, zero_f)
    denom = ct.sum(numer_masked, axis=1, keepdims=True)                # (BLOCK_M, 1)
    probs = numer_masked / denom

    raw_max_1d = ct.reshape(raw_max, (BLOCK_M_,))
    scaled_max_1d = ct.reshape(scaled_max, (BLOCK_M_,))
    all_finite_1d = ct.reshape(all_finite, (BLOCK_M_,))
    denom_1d = ct.reshape(denom, (BLOCK_M_,))

    # Row-major flat indices into the compact side-output arrays.
    ct.scatter(amax_ptr, row_offs, raw_max_1d, mask=row_mask_1d)
    ct.scatter(scaled_amax_ptr, row_offs, scaled_max_1d, mask=row_mask_1d)
    ct.scatter(all_finite_ptr, row_offs, all_finite_1d, mask=row_mask_1d)
    ct.scatter(denom_ptr, row_offs, denom_1d, mask=row_mask_1d)

    # Store probs into flat [N_HEADS*K_LEN*K_LEN].
    probs_bf = ct.astype(probs, ct.bfloat16)
    out_off = ct.reshape(row_offs, (BLOCK_M_, 1)) * K_LEN_ + ct.reshape(cols, (1, BLOCK_N_))
    ct.scatter(out_flat_ptr, out_off, probs_bf, mask=valid_2d)


@oracle_impl(hardware="B200", point="866d908a")
def oracle_forward(inputs):
    arg0_1, _shape_param_0, _shape_param_1, _shape_param_2 = inputs
    out_shape = tuple(int(dim) for dim in _shape_param_2)
    device = arg0_1.device

    reduction_shape = (128, 6, 49, 1)
    reduction_stride = (294, 49, 1, 1)
    amax = torch.empty_strided(reduction_shape, reduction_stride, device=device, dtype=torch.float32)
    scaled_amax = torch.empty_strided(reduction_shape, reduction_stride, device=device, dtype=torch.float32)
    all_finite = torch.empty_strided(reduction_shape, reduction_stride, device=device, dtype=torch.bool)
    denom = torch.empty_strided(reduction_shape, reduction_stride, device=device, dtype=torch.float32)
    out = torch.empty_strided(
        out_shape,
        (out_shape[1] * out_shape[2], out_shape[2], 1),
        device=device,
        dtype=torch.bfloat16,
    )

    # Flatten side outputs to a 1D view of length 128*6*49.
    amax_flat = amax.view(-1)
    scaled_amax_flat = scaled_amax.view(-1)
    all_finite_flat = all_finite.view(-1)
    denom_flat = denom.view(-1)
    out_flat = out.view(-1)
    x_flat = arg0_1.view(-1)  # contiguous view of bf16[768*56*56]

    n_rows = N_HEADS * K_LEN
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(n_rows, BLOCK_M), 1, 1),
        _visformer_safe_softmax_kernel,
        (
            x_flat, amax_flat, scaled_amax_flat, all_finite_flat, denom_flat, out_flat,
            n_rows, K_LEN, INPUT_H, BLOCK_M, BLOCK_N, SCALE,
        ),
    )
    return amax, scaled_amax, all_finite, denom, out, out.permute(0, 2, 1)
