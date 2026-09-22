"""cuTile port of sum_sum_sum_2261b2f5694a: MobileBERT bf16 LayerNorm-backward
6-output return scope.

Loads ROWS_PER_GROUP rows per block as a single tile (no inner Python loop),
computes the per-row grad, masked side-output, and column partials.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


ROWS = 32768
HIDDEN = 512
ROW_FACTOR = 512.0
INV_ROW_FACTOR = 1.0 / 512.0


@ct.kernel
def _row_partials_kernel(
    x_ptr,             # bf16 [ROWS, HIDDEN]
    weight_ptr,        # f32 [HIDDEN]
    relu_input_ptr,    # bf16 [ROWS, HIDDEN]
    center_ptr,        # f32 [ROWS]
    scale_ptr,         # f32 [ROWS]
    masked_out_ptr,    # bf16 [ROWS, HIDDEN]
    partials_ptr,      # f32 [NUM_GROUPS, 3, HIDDEN]
    ROWS_PER_GROUP: ct.Constant[int],
    HIDDEN_: ct.Constant[int],
):
    group = ct.bid(0)

    weight = ct.load(weight_ptr, index=(0,), shape=(HIDDEN_,))
    weight_2d = ct.reshape(weight, (1, HIDDEN_))

    # Load the full BLOCK_R = ROWS_PER_GROUP rows at once.
    x_bf16 = ct.load(x_ptr, index=(group, 0),
                     shape=(ROWS_PER_GROUP, HIDDEN_))
    relu_input_bf16 = ct.load(relu_input_ptr, index=(group, 0),
                              shape=(ROWS_PER_GROUP, HIDDEN_))
    center = ct.load(center_ptr, index=(group,), shape=(ROWS_PER_GROUP,))
    scale = ct.load(scale_ptr, index=(group,), shape=(ROWS_PER_GROUP,))

    x = ct.astype(x_bf16, ct.float32)
    relu_input = ct.astype(relu_input_bf16, ct.float32)
    center_2d = ct.reshape(center, (ROWS_PER_GROUP, 1))
    scale_2d = ct.reshape(scale, (ROWS_PER_GROUP, 1))

    relu = ct.where(relu_input > 0.0, relu_input,
                    ct.zeros((ROWS_PER_GROUP, HIDDEN_), dtype=ct.float32))
    norm = (relu - center_2d) * scale_2d
    weighted = x * weight_2d
    row_sum = ct.sum(weighted, axis=1, keepdims=True)
    row_dot = ct.sum(weighted * norm, axis=1, keepdims=True)

    centered = weighted * ROW_FACTOR - row_sum - norm * row_dot
    grad = scale_2d * INV_ROW_FACTOR * centered
    grad_bf16 = ct.astype(grad, ct.bfloat16)
    grad_bf16_f32 = ct.astype(grad_bf16, ct.float32)
    masked_f32 = ct.where(relu <= 0.0,
                          ct.zeros((ROWS_PER_GROUP, HIDDEN_), dtype=ct.float32),
                          grad_bf16_f32)
    masked_bf16 = ct.astype(masked_f32, ct.bfloat16)
    ct.store(masked_out_ptr, index=(group, 0), tile=masked_bf16)

    # Column partial sums
    xn_sum = ct.sum(x * norm, axis=0)
    x_sum = ct.sum(x, axis=0)
    m_sum = ct.sum(ct.astype(masked_bf16, ct.float32), axis=0)

    ct.store(partials_ptr, index=(group, 0, 0),
             tile=ct.reshape(xn_sum, (1, 1, HIDDEN_)))
    ct.store(partials_ptr, index=(group, 1, 0),
             tile=ct.reshape(x_sum, (1, 1, HIDDEN_)))
    ct.store(partials_ptr, index=(group, 2, 0),
             tile=ct.reshape(m_sum, (1, 1, HIDDEN_)))


@ct.kernel
def _finalize_partials_kernel(
    partials_ptr,        # f32 [NUM_GROUPS, 3, HIDDEN]
    out_x_norm_ptr,      # f32 [HIDDEN]
    out_x_ptr,           # f32 [HIDDEN]
    out_masked_sum_ptr,  # f32 [HIDDEN]
    NUM_GROUPS: ct.Constant[int],
    HIDDEN_: ct.Constant[int],
    FINAL_BLOCK_C: ct.Constant[int],
):
    c_tile = ct.bid(0)

    xn = ct.load(partials_ptr, index=(0, 0, c_tile),
                 shape=(NUM_GROUPS, 1, FINAL_BLOCK_C))
    x = ct.load(partials_ptr, index=(0, 1, c_tile),
                shape=(NUM_GROUPS, 1, FINAL_BLOCK_C))
    m = ct.load(partials_ptr, index=(0, 2, c_tile),
                shape=(NUM_GROUPS, 1, FINAL_BLOCK_C))
    acc_xn = ct.reshape(ct.sum(xn, axis=0), (FINAL_BLOCK_C,))
    acc_x = ct.reshape(ct.sum(x, axis=0), (FINAL_BLOCK_C,))
    acc_m = ct.reshape(ct.sum(m, axis=0), (FINAL_BLOCK_C,))

    ct.store(out_x_norm_ptr, index=(c_tile,), tile=acc_xn)
    ct.store(out_x_ptr, index=(c_tile,), tile=acc_x)
    m_bf = ct.astype(acc_m, ct.bfloat16)
    ct.store(out_masked_sum_ptr, index=(c_tile,), tile=ct.astype(m_bf, ct.float32))


def _shape_tuple(shape):
    return tuple(int(dim) for dim in shape)


def _next_pow2(v):
    r = 1
    while r < v:
        r <<= 1
    return r


@oracle_impl(
    hardware="B200",
    point="81eeb2b9",
    ROWS_PER_GROUP=16,
    FINAL_BLOCK_C=8,
)
def oracle_forward(inputs, *, ROWS_PER_GROUP, FINAL_BLOCK_C):
    (
        x_bf16,
        weight,
        relu_input_bf16,
        center,
        scale,
        _shape0,
        _shape1,
        masked_shape_param,
        sum_shape_param,
    ) = inputs
    masked_shape = _shape_tuple(masked_shape_param)
    sum_shape = _shape_tuple(sum_shape_param)
    rows = int(x_bf16.shape[0])
    hidden = int(x_bf16.shape[1])
    assert rows == ROWS and hidden == HIDDEN

    center_1d = center.view(rows)
    scale_1d = scale.view(rows)

    full = torch.zeros((), device=x_bf16.device, dtype=torch.bfloat16)
    masked = torch.empty_strided(
        masked_shape,
        (hidden, 1),
        device=x_bf16.device,
        dtype=torch.bfloat16,
    )
    masked_2d = masked.view(rows, hidden)

    out_x_norm = torch.empty_strided(sum_shape, (1,), device=x_bf16.device, dtype=torch.float32)
    out_x = torch.empty_strided(sum_shape, (1,), device=x_bf16.device, dtype=torch.float32)
    out_masked_sum = torch.empty_strided(sum_shape, (1,), device=x_bf16.device, dtype=torch.float32)

    num_groups = (rows + ROWS_PER_GROUP - 1) // ROWS_PER_GROUP
    partials = torch.empty(
        (num_groups, 3, hidden),
        device=x_bf16.device,
        dtype=torch.float32,
    )

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (num_groups, 1, 1),
        _row_partials_kernel,
        (x_bf16, weight, relu_input_bf16, center_1d, scale_1d,
         masked_2d, partials, ROWS_PER_GROUP, hidden),
    )

    num_groups_pow2 = _next_pow2(num_groups)
    if num_groups_pow2 != num_groups:
        partials_pad = torch.zeros((num_groups_pow2, 3, hidden), device=x_bf16.device,
                                    dtype=torch.float32)
        partials_pad[:num_groups] = partials
        partials = partials_pad

    ct.launch(
        stream,
        (hidden // FINAL_BLOCK_C, 1, 1),
        _finalize_partials_kernel,
        (partials, out_x_norm, out_x, out_masked_sum,
         num_groups_pow2, hidden, FINAL_BLOCK_C),
    )

    return out_x_norm, out_x, full, masked, masked.permute(1, 0), out_masked_sum
