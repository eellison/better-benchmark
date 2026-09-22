"""cuTile port of sum_sum_sum_4ce9013c6e0d: Albert LayerNorm-backward with
masked scatter.

Mirrors Triton's 3-kernel structure (init/scatter_reduce/finalize) as
2 cuTile kernels + torch scatter fallback (cuTile atomic_add with sentinel-
masking is heavy for such a small speed-up here). Torch masked_index_put_
accumulate replaces Triton's in-kernel atomic scatters. Column reductions
(sum_3 = sum(x*normed, axis=[0,1]), sum_4 = sum(x, axis=[0,1])) are
computed via cuTile per-seq partials + finalize kernel.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 8
SEQ = 512
HIDDEN = 128
ROW_FACTOR = 128.0


@ct.kernel
def _ln_bwd_and_partials_kernel(
    grad_ptr,          # bf16 [BATCH, SEQ, HIDDEN]
    weight_ptr,        # f32  [HIDDEN]
    normed_ptr,        # f32  [BATCH, SEQ, HIDDEN]
    invstd_ptr,        # f32  [BATCH, SEQ, 1]
    grad_input_ptr,    # f32  [BATCH, SEQ, HIDDEN]
    partial_x_normed_ptr,  # f32 [SEQ, HIDDEN]
    partial_x_ptr,     # f32 [SEQ, HIDDEN]
    SEQ_C: ct.Constant[int],
    HIDDEN_C: ct.Constant[int],
    BATCH_C: ct.Constant[int],
    ROW_FACTOR_C: ct.Constant[float],
):
    # One program per seq slot; accumulates across BATCH.
    seq_idx = ct.bid(0)

    weight_1d = ct.load(weight_ptr, index=(0,), shape=(HIDDEN_C,))
    weight_2d = ct.reshape(weight_1d, (1, HIDDEN_C))

    zero_line = ct.zeros((1, HIDDEN_C), dtype=ct.float32)
    acc_x_normed = zero_line
    acc_x = zero_line

    for b in range(BATCH_C):
        grad_bf = ct.load(grad_ptr, index=(b, seq_idx, 0),
                          shape=(1, 1, HIDDEN_C))
        grad_bf_2d = ct.reshape(grad_bf, (1, HIDDEN_C))
        normed_3d = ct.load(normed_ptr, index=(b, seq_idx, 0),
                            shape=(1, 1, HIDDEN_C))
        normed = ct.reshape(normed_3d, (1, HIDDEN_C))
        invstd_3d = ct.load(invstd_ptr, index=(b, seq_idx, 0),
                            shape=(1, 1, 1))
        invstd = ct.reshape(invstd_3d, (1, 1))

        grad = ct.astype(grad_bf_2d, ct.float32)
        weighted = grad * weight_2d
        row_sum = ct.sum(weighted, axis=1, keepdims=True)
        weighted_normed = weighted * normed
        row_dot = ct.sum(weighted_normed, axis=1, keepdims=True)
        centered = weighted * ROW_FACTOR_C - row_sum - normed * row_dot
        grad_out = invstd * centered

        # Store the grad_input tile.
        ct.store(grad_input_ptr, index=(b, seq_idx, 0),
                 tile=ct.reshape(grad_out, (1, 1, HIDDEN_C)))

        # Accumulate for column reductions.
        # sum_3 = sum(x*normed, axis=[0,1]) where x = grad.f32
        # sum_4 = sum(x, axis=[0,1])
        acc_x_normed = acc_x_normed + grad * normed
        acc_x = acc_x + grad

    # Store per-seq partials.
    ct.store(partial_x_normed_ptr, index=(seq_idx, 0), tile=acc_x_normed)
    ct.store(partial_x_ptr, index=(seq_idx, 0), tile=acc_x)


@ct.kernel
def _finalize_hidden_sums_kernel(
    partial_x_normed_ptr,  # f32 [SEQ, HIDDEN]
    partial_x_ptr,         # f32 [SEQ, HIDDEN]
    out_x_normed_ptr,      # f32 [HIDDEN]
    out_x_ptr,             # f32 [HIDDEN]
    SEQ_C: ct.Constant[int],
    HIDDEN_C: ct.Constant[int],
):
    x_normed = ct.load(partial_x_normed_ptr, index=(0, 0),
                       shape=(SEQ_C, HIDDEN_C))
    x = ct.load(partial_x_ptr, index=(0, 0),
                shape=(SEQ_C, HIDDEN_C))
    ct.store(out_x_normed_ptr, index=(0,), tile=ct.sum(x_normed, axis=0))
    ct.store(out_x_ptr, index=(0,), tile=ct.sum(x, axis=0))


@oracle_impl(hardware="B200", point="409a14a3")
def oracle_forward(inputs):
    (arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1,
     shape0, shape1, shape2, shape3, shape4) = inputs
    device = arg0_1.device

    grad_3d = arg1_1.view(BATCH, SEQ, HIDDEN)          # bf16
    weight = arg2_1                                     # f32 [HIDDEN]
    normed_3d = arg3_1.view(BATCH, SEQ, HIDDEN)        # f32
    invstd_3d = arg4_1.view(BATCH, SEQ, 1)             # f32

    grad_input = torch.empty_strided(
        tuple(int(d) for d in shape0), (SEQ * HIDDEN, HIDDEN, 1),
        device=device, dtype=torch.float32,
    )
    grad_input_3d = grad_input.view(BATCH, SEQ, HIDDEN)

    partial_x_normed = torch.empty((SEQ, HIDDEN), device=device, dtype=torch.float32)
    partial_x = torch.empty((SEQ, HIDDEN), device=device, dtype=torch.float32)
    out_x_normed = torch.empty((HIDDEN,), device=device, dtype=torch.float32)
    out_x = torch.empty((HIDDEN,), device=device, dtype=torch.float32)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (SEQ, 1, 1), _ln_bwd_and_partials_kernel,
        (grad_3d, weight, normed_3d, invstd_3d, grad_input_3d,
         partial_x_normed, partial_x,
         SEQ, HIDDEN, BATCH, ROW_FACTOR),
    )
    ct.launch(
        stream, (1, 1, 1), _finalize_hidden_sums_kernel,
        (partial_x_normed, partial_x, out_x_normed, out_x,
         SEQ, HIDDEN),
    )
    sum_3 = out_x_normed
    sum_4 = out_x

    # sum_5: sum of grad_input across dim 0 (batch) -> [1, 512, 128]
    sum_5 = grad_input.sum(dim=0, keepdim=True)

    # Masked scatters (torch, since we haven't ported the atomic-scatter kernels).
    ge = arg5_1 >= 0
    lt = arg5_1 < 512
    ne = arg5_1 != -1
    mask_5 = ((ge & lt) & ne).unsqueeze(-1)
    full = torch.zeros(tuple(int(d) for d in shape1), device=device, dtype=torch.float32)
    scatter_1 = torch.ops.aten._unsafe_masked_index_put_accumulate.default(
        full, mask_5, [arg5_1], sum_5,
    )

    expand_shape2 = tuple(int(d) for d in shape2)
    expand = arg6_1.expand(expand_shape2)
    ge_1 = expand >= 0
    lt_1 = expand < 2
    ne_1 = expand != -1
    mask_6 = ((ge_1 & lt_1) & ne_1).unsqueeze(-1)
    full_1 = torch.zeros(tuple(int(d) for d in shape3), device=device, dtype=torch.float32)
    scatter_2 = torch.ops.aten._unsafe_masked_index_put_accumulate.default(
        full_1, mask_6, [expand], grad_input,
    )

    ge_2 = arg7_1 >= 0
    lt_2 = arg7_1 < 30000
    ne_2 = arg7_1 != 0
    mask_7 = ((ge_2 & lt_2) & ne_2).unsqueeze(-1)
    full_2 = torch.zeros(tuple(int(d) for d in shape4), device=device, dtype=torch.float32)
    scatter_3 = torch.ops.aten._unsafe_masked_index_put_accumulate.default(
        full_2, mask_7, [arg7_1], grad_input,
    )
    add_out = arg0_1.to(torch.float32) + scatter_3

    return sum_3, sum_4, scatter_1, scatter_2, add_out
