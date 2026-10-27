"""cuTile port of sum_sum_sum_86e661f4be59: GhostNet dual BN-backward.

Two BN-backward pipelines: one for 112 channels (14x14 spatial) and one for
56 channels sliced from the same buffer. cuTile kernels do the channel
reductions (sum_x and sum_dot per channel) in-kernel — matching Triton's
_materialize_and_partial_kernel + _finalize_*_kernel structure — and the
BN-backward per-element epilogues.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


N = 512
H = 14
W = 14
HW = H * W
C1 = 112
C2 = 56
NUMEL1 = N * C1 * HW
NUMEL2 = N * C2 * HW
INV_R = 9.964923469387754e-06
BLOCK = 1024
K_TOTAL = N * HW


def _pow2_ceil(x: int) -> int:
    return 1 << (int(x) - 1).bit_length()


@ct.kernel
def _col_partials_kernel(
    value_ptr,      # f32 [rows, C]
    centered_ptr,   # f32 [rows, C]
    partial_x_ptr,  # f32 [rows, C]
    partial_dot_ptr,# f32 [rows, C]
    C_: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    row = ct.bid(0)
    col_block = ct.bid(1)
    v = ct.load(value_ptr, index=(row, col_block), shape=(1, BLOCK_C))
    ce = ct.load(centered_ptr, index=(row, col_block), shape=(1, BLOCK_C))
    ct.store(partial_x_ptr, index=(row, col_block), tile=v)
    ct.store(partial_dot_ptr, index=(row, col_block), tile=v * ce)


@ct.kernel
def _finalize_col_kernel(
    partial_x_ptr,   # f32 [rows, C]
    partial_dot_ptr, # f32 [rows, C]
    out_sum_x_ptr,   # f32 [C]
    out_sum_dot_ptr, # f32 [C]
    ROWS: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    col_block = ct.bid(0)
    acc_x = ct.zeros((BLOCK_C,), dtype=ct.float32)
    acc_dot = ct.zeros((BLOCK_C,), dtype=ct.float32)
    for r in range(ROWS):
        xv = ct.load(partial_x_ptr, index=(r, col_block), shape=(1, BLOCK_C))
        dv = ct.load(partial_dot_ptr, index=(r, col_block), shape=(1, BLOCK_C))
        acc_x = acc_x + ct.reshape(xv, (BLOCK_C,))
        acc_dot = acc_dot + ct.reshape(dv, (BLOCK_C,))
    ct.store(out_sum_x_ptr, index=(col_block,), tile=acc_x)
    ct.store(out_sum_dot_ptr, index=(col_block,), tile=acc_dot)


@ct.kernel
def _bn_backward_epilogue_kernel(
    value_ptr,         # f32 [NUMEL]
    centered_ptr,      # f32 [NUMEL]
    mean_term_ptr,     # f32 [C]
    dot_coeff_ptr,     # f32 [C]
    out_scale_ptr,     # f32 [C]
    channel_of_ptr,    # i32 [NUMEL]
    out_ptr,           # bf16 [NUMEL]
    BLOCK_: ct.Constant[int],
):
    pid = ct.bid(0)
    value = ct.load(value_ptr, index=(pid,), shape=(BLOCK_,))
    centered = ct.load(centered_ptr, index=(pid,), shape=(BLOCK_,))
    ch = ct.load(channel_of_ptr, index=(pid,), shape=(BLOCK_,))
    mt = ct.gather(mean_term_ptr, ch)
    dc = ct.gather(dot_coeff_ptr, ch)
    osv = ct.gather(out_scale_ptr, ch)
    adjusted = value - centered * dc - mt
    out = adjusted * osv
    ct.store(out_ptr, index=(pid,), tile=ct.astype(out, ct.bfloat16))


def _bn_backward(value_f32, centered, weight, bias, sum_1, sum_2, C, NUMEL, device):
    mean_term = sum_1 * INV_R
    dot_coeff = (sum_2 * INV_R) * (weight * weight)
    out_scale = weight * bias

    value_flat = value_f32.contiguous().view(-1)
    centered_flat = centered.contiguous().view(-1)

    n_range = torch.arange(NUMEL, device=device, dtype=torch.int32)
    channel_of = (n_range // HW) % C

    out_flat = torch.empty((NUMEL,), device=device, dtype=torch.bfloat16)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (NUMEL // BLOCK, 1, 1), _bn_backward_epilogue_kernel,
        (value_flat, centered_flat, mean_term, dot_coeff, out_scale,
         channel_of, out_flat, BLOCK),
    )
    dense_out = torch.empty_strided(
        (N, C, H, W), (C * HW, 1, W * C, C),
        device=device, dtype=torch.bfloat16,
    )
    dense_out.copy_(out_flat.view(N, C, H, W))
    return dense_out


def _col_reduce(value_f32, centered, C, K_ROWS, device, block_c=8):
    """Compute sum_1 = value.sum(dim=[0,2,3]) and sum_2 = (value*centered).sum(dim=[0,2,3]).
    We first flatten value/centered to (K_ROWS, C) with channels-last layout,
    then do two cuTile kernels: partial (which just copies value and value*centered
    into partials aligned to (rows, C) layout) then finalize.

    Note: this is a lightweight structural fix — the col reductions live in-kernel.
    """
    # value_f32 is (N, C, H, W) contiguous — reshape to (N*HW, C) via permute
    # value_f32 stride is (C*HW, HW, W, 1). We need (K_ROWS, C).
    # Rather than a costly permute, use the identity:
    #   sum(dim=[0,2,3]) over (N,C,H,W) contig == sum over rows of transposed.
    # We use torch here for the layout reshape (view-only for channels-last input)
    # then feed the (K_ROWS, C) view to cuTile.
    value_2d = value_f32.permute(0, 2, 3, 1).contiguous().view(K_ROWS, C)
    centered_2d = centered.permute(0, 2, 3, 1).contiguous().view(K_ROWS, C)
    partial_x = torch.empty((K_ROWS, C), device=device, dtype=torch.float32)
    partial_dot = torch.empty((K_ROWS, C), device=device, dtype=torch.float32)
    out_x = torch.empty((C,), device=device, dtype=torch.float32)
    out_dot = torch.empty((C,), device=device, dtype=torch.float32)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (K_ROWS, C // block_c, 1), _col_partials_kernel,
        (value_2d, centered_2d, partial_x, partial_dot, C, block_c),
    )
    ct.launch(
        stream, (C // block_c, 1, 1), _finalize_col_kernel,
        (partial_x, partial_dot, out_x, out_dot, K_ROWS, block_c),
    )
    return out_x, out_dot


@oracle_impl(hardware="B200", point="641bb11a")
def oracle_forward(inputs):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, shape_p0, shape_p1 = inputs
    device = arg0_1.device

    add_bf = arg0_1 + arg1_1
    _shape_p0 = tuple(int(x) for x in shape_p0)
    _shape_p1 = tuple(int(x) for x in shape_p1)
    new_empty = torch.empty_strided(_shape_p0, _shape_p1, device=device, dtype=torch.bfloat16)
    new_empty.copy_(add_bf)
    clone = new_empty.clone(memory_format=torch.contiguous_format)
    copy_1 = torch.ops.aten.copy.default(new_empty, clone)

    # First BN backward (C=112). arg3_1 shape is (1, C, 1, 1).
    value_f32 = clone.to(torch.float32)
    centered = arg2_1.to(torch.float32) - arg3_1
    sum_1, sum_2 = _col_reduce(value_f32, centered, C1, N * HW, device)
    dense1 = _bn_backward(value_f32, centered, arg4_1, arg5_1, sum_1, sum_2, C1, NUMEL1, device)
    mul_8 = sum_2 * arg4_1

    # Second BN backward (C=56, slice[:, 56:112])
    slice_1 = copy_1[:, 56:112]
    value_f32_2 = slice_1.to(torch.float32)
    centered_2 = arg6_1.to(torch.float32) - arg7_1
    sum_3, sum_4 = _col_reduce(value_f32_2, centered_2, C2, N * HW, device)
    dense2 = _bn_backward(value_f32_2, centered_2, arg8_1, arg9_1, sum_3, sum_4, C2, NUMEL2, device)
    mul_17 = sum_4 * arg8_1

    return copy_1, sum_1, mul_8, dense1, sum_3, mul_17, dense2
