"""cuTile port of sum_sum_f2a80116b65b: GhostNet hard-sigmoid producer + BN-backward.

Reference (channels-last strided inputs; channel innermost after NHWC permute):
  gate      = f32(arg0)   [N, C, 1, 1]
  gate_s    = clamp(gate + 3, 0, 6) / 6
  gate_bf   = bf16(gate_s)                             [N, C, 1, 1]
  mul       = arg1 * gate_bf                            bf16 [N, C, H, W]
  div       = bf16(arg2 / 784)                          bf16 [N, C, 1, 1]
  add       = mul + div                                 bf16 [N, C, H, W]  (broadcast)
  value_f   = f32(add)
  sum_1     = sum(value_f, [0, 2, 3])                   f32 [C]
  centered  = f32(arg4) - arg3.squeeze                   f32 [N, C, H, W]
  sum_2     = sum(value_f * centered, [0, 2, 3])       f32 [C]
  variance  = sum_2 * SCALE * invstd^2                   (SCALE = 2.4912308673469386e-06)
  mean_term = sum_1 * SCALE
  output_scale = invstd * weight
  grad      = bf16((value_f - centered*variance - mean_term) * output_scale)
                                                          bf16 [N, C, H, W] cl
  scale_grad = sum_2 * invstd
  Returns (sum_1, scale_grad, grad).

Port strategy: torch handles the elementwise producer (fast on bf16). Two
cuTile kernels: partial reduce + BN-backward writeback. Column dim C is
non-power-of-2 (72 or 672), so we pad to next_pow2 in the tile and mask
inactive columns. C, H, W come from shapes.json:
  * 6b37439d: (C=72, H=W=28)  -> BLOCK_C=128
  * 84fdc974: (C=672, H=W=7)  -> BLOCK_C=1024
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


SCALE = 2.4912308673469386e-06


def _make_kernels(BLOCK_ROW, BLOCK_C):

    @ct.kernel
    def _partial_reduce(
        add_ptr,           # bf16 [ROWS, BLOCK_C] (padded)
        centered_ptr,      # f32  [ROWS, BLOCK_C]
        partial_sum_ptr,   # f32  [NUM_BLOCKS, BLOCK_C]
        partial_dot_ptr,   # f32  [NUM_BLOCKS, BLOCK_C]
        C_: ct.Constant[int],
        ROWS_: ct.Constant[int],
    ):
        r = ct.bid(0)
        add_v = ct.load(
            add_ptr, index=(r, 0), shape=(BLOCK_ROW, BLOCK_C),
            padding_mode=ct.PaddingMode.ZERO,
        )
        centered = ct.load(
            centered_ptr, index=(r, 0), shape=(BLOCK_ROW, BLOCK_C),
            padding_mode=ct.PaddingMode.ZERO,
        )

        cols = ct.arange(BLOCK_C, dtype=ct.int32)
        rows = ct.arange(BLOCK_ROW, dtype=ct.int32)
        col_mask = ct.reshape(cols < C_, (1, BLOCK_C))
        row_off = r * BLOCK_ROW + rows
        row_mask = ct.reshape(row_off < ROWS_, (BLOCK_ROW, 1))
        active = ct.broadcast_to(col_mask, (BLOCK_ROW, BLOCK_C)) & \
                 ct.broadcast_to(row_mask, (BLOCK_ROW, BLOCK_C))

        add_f = ct.astype(add_v, ct.float32)
        add_f = ct.where(active, add_f, 0.0)
        centered = ct.where(active, centered, 0.0)

        partial_sum = ct.sum(add_f, axis=0)                # (BLOCK_C,)
        partial_dot = ct.sum(add_f * centered, axis=0)     # (BLOCK_C,)
        ct.store(partial_sum_ptr, index=(r, 0),
                 tile=ct.reshape(partial_sum, (1, BLOCK_C)))
        ct.store(partial_dot_ptr, index=(r, 0),
                 tile=ct.reshape(partial_dot, (1, BLOCK_C)))

    @ct.kernel
    def _epilogue(
        add_ptr,           # bf16 [ROWS, BLOCK_C] padded
        centered_ptr,      # f32  [ROWS, BLOCK_C]
        mean_term_ptr,     # f32  [BLOCK_C] padded
        variance_ptr,      # f32  [BLOCK_C]
        output_scale_ptr,  # f32  [BLOCK_C]
        out_ptr,           # bf16 [ROWS, BLOCK_C] padded
        C_: ct.Constant[int],
        ROWS_: ct.Constant[int],
    ):
        r = ct.bid(0)
        add_v = ct.load(
            add_ptr, index=(r, 0), shape=(BLOCK_ROW, BLOCK_C),
            padding_mode=ct.PaddingMode.ZERO,
        )
        centered = ct.load(
            centered_ptr, index=(r, 0), shape=(BLOCK_ROW, BLOCK_C),
            padding_mode=ct.PaddingMode.ZERO,
        )
        mean_term = ct.load(mean_term_ptr, index=(0,), shape=(BLOCK_C,))
        variance = ct.load(variance_ptr, index=(0,), shape=(BLOCK_C,))
        output_scale = ct.load(output_scale_ptr, index=(0,), shape=(BLOCK_C,))

        add_f = ct.astype(add_v, ct.float32)
        mean_term_bc = ct.broadcast_to(ct.reshape(mean_term, (1, BLOCK_C)),
                                        (BLOCK_ROW, BLOCK_C))
        variance_bc = ct.broadcast_to(ct.reshape(variance, (1, BLOCK_C)),
                                       (BLOCK_ROW, BLOCK_C))
        out_scale_bc = ct.broadcast_to(ct.reshape(output_scale, (1, BLOCK_C)),
                                        (BLOCK_ROW, BLOCK_C))
        after_variance = add_f - centered * variance_bc
        after_mean = after_variance - mean_term_bc
        out_f = after_mean * out_scale_bc
        out_bf = ct.astype(out_f, ct.bfloat16)
        ct.store(out_ptr, index=(r, 0), tile=out_bf)

    return _partial_reduce, _epilogue


_K_72_reduce, _K_72_epilogue = _make_kernels(BLOCK_ROW=128, BLOCK_C=128)
# For C=672 use narrower BLOCK_ROW to keep tile size sane.
_K_672_reduce, _K_672_epilogue = _make_kernels(BLOCK_ROW=64, BLOCK_C=1024)


def _run(inputs, *, C, H, W, BLOCK_ROW, BLOCK_C, reduce_k, epilogue_k):
    gate, source, bias, mean, bn_input, invstd, weight, _shape = inputs
    device = source.device
    N = int(source.shape[0])
    HW = H * W
    ROWS = N * HW
    NUM_BLOCKS = (ROWS + BLOCK_ROW - 1) // BLOCK_ROW
    ROWS_PAD = NUM_BLOCKS * BLOCK_ROW

    # Torch elementwise producer (matches captured hard-sigmoid gate).
    gate_f = gate.to(torch.float32)                           # [N, C, 1, 1]
    gate_s = torch.clamp(gate_f + 3, 0, 6) / 6
    gate_bf = gate_s.to(torch.bfloat16)                       # [N, C, 1, 1]
    mul = source * gate_bf                                     # bf16 [N, C, H, W]
    div = (bias.to(torch.float32) / 784.0).to(torch.bfloat16)
    add = mul + div                                            # bf16 [N, C, H, W]

    # centered = f32(bn_input) - mean.squeeze
    mean_1d = mean.view(C)
    bn_input_f = bn_input.to(torch.float32)
    centered = bn_input_f - mean_1d.view(1, C, 1, 1)          # f32 [N, C, H, W]

    # Permute to NHWC contiguous, pad columns to BLOCK_C.
    add_nhwc = add.permute(0, 2, 3, 1).contiguous()            # [N, H, W, C]
    centered_nhwc = centered.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]

    add_pad = torch.zeros((ROWS_PAD, BLOCK_C), device=device, dtype=torch.bfloat16)
    add_pad[:ROWS, :C] = add_nhwc.view(ROWS, C)
    centered_pad = torch.zeros((ROWS_PAD, BLOCK_C), device=device, dtype=torch.float32)
    centered_pad[:ROWS, :C] = centered_nhwc.view(ROWS, C)

    partial_sum = torch.empty((NUM_BLOCKS, BLOCK_C), device=device, dtype=torch.float32)
    partial_dot = torch.empty((NUM_BLOCKS, BLOCK_C), device=device, dtype=torch.float32)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (NUM_BLOCKS, 1, 1), reduce_k,
        (add_pad, centered_pad, partial_sum, partial_dot, C, ROWS),
    )

    sum_out = partial_sum[:, :C].sum(dim=0)
    dot_out = partial_dot[:, :C].sum(dim=0)
    invstd_1d = invstd.view(C)
    weight_1d = weight.view(C) if weight.ndim > 1 else weight
    scale_grad = dot_out * invstd_1d
    mean_term = sum_out * SCALE
    variance = dot_out * SCALE * (invstd_1d * invstd_1d)
    output_scale = invstd_1d * weight_1d

    mean_term_pad = torch.zeros((BLOCK_C,), device=device, dtype=torch.float32)
    mean_term_pad[:C] = mean_term
    variance_pad = torch.zeros((BLOCK_C,), device=device, dtype=torch.float32)
    variance_pad[:C] = variance
    output_scale_pad = torch.zeros((BLOCK_C,), device=device, dtype=torch.float32)
    output_scale_pad[:C] = output_scale

    out_pad = torch.empty((ROWS_PAD, BLOCK_C), device=device, dtype=torch.bfloat16)

    ct.launch(
        stream, (NUM_BLOCKS, 1, 1), epilogue_k,
        (add_pad, centered_pad, mean_term_pad, variance_pad, output_scale_pad,
         out_pad, C, ROWS),
    )

    out_nhwc = out_pad[:ROWS, :C].contiguous().view(N, H, W, C)
    out_cl = torch.empty_strided(
        (N, C, H, W),
        (C * HW, 1, W * C, C),
        device=device, dtype=torch.bfloat16,
    )
    out_cl.copy_(out_nhwc.permute(0, 3, 1, 2))

    return sum_out, scale_grad, out_cl


@oracle_impl(hardware="B200", point="6b37439d",
             C=72, H=28, W=28, BLOCK_ROW=128, BLOCK_C=128, variant=72)
@oracle_impl(hardware="B200", point="84fdc974",
             C=672, H=7, W=7, BLOCK_ROW=64, BLOCK_C=1024, variant=672)
def oracle_forward(inputs, *, C, H, W, BLOCK_ROW, BLOCK_C, variant):
    if variant == 72:
        reduce_k, epilogue_k = _K_72_reduce, _K_72_epilogue
    else:
        reduce_k, epilogue_k = _K_672_reduce, _K_672_epilogue
    return _run(inputs, C=C, H=H, W=W, BLOCK_ROW=BLOCK_ROW, BLOCK_C=BLOCK_C,
                reduce_k=reduce_k, epilogue_k=epilogue_k)
