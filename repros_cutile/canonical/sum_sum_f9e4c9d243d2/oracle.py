"""cuTile port of sum_sum_f9e4c9d243d2: ShuffleNet channel-shuffle + BN
backward tail.

Matches Triton's 3-kernel structure by moving the two sum reductions into
cuTile kernels (partial + finalize) alongside the existing BN-backward
elementwise epilogue.

Shuffle/BN-forward/where handled in torch (matches the layout complexity —
Triton's kernel does channels-last loads/stores which are handled here via
torch's contiguous permutations).
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


SCALE = 3.985969387755102e-05


@ct.kernel
def _partial_reduce_kernel(
    producer_ptr,      # f32 (n, C, hw)
    centered_ptr,      # f32 (n, C, hw)
    partial_sum_ptr,   # f32 (n, C)
    partial_dot_ptr,   # f32 (n, C)
    HW_PAD: ct.Constant[int],
):
    n = ct.bid(0)
    c = ct.bid(1)
    producer = ct.load(producer_ptr, index=(n, c, 0), shape=(1, 1, HW_PAD),
                       padding_mode=ct.PaddingMode.ZERO)
    centered = ct.load(centered_ptr, index=(n, c, 0), shape=(1, 1, HW_PAD),
                       padding_mode=ct.PaddingMode.ZERO)
    producer_2d = ct.reshape(producer, (1, HW_PAD))
    centered_2d = ct.reshape(centered, (1, HW_PAD))
    sum_v = ct.sum(producer_2d, axis=None)  # scalar
    dot_v = ct.sum(producer_2d * centered_2d, axis=None)  # scalar
    ct.store(partial_sum_ptr, index=(n, c), tile=ct.reshape(sum_v, (1, 1)))
    ct.store(partial_dot_ptr, index=(n, c), tile=ct.reshape(dot_v, (1, 1)))


@ct.kernel
def _finalize_kernel(
    partial_sum_ptr,     # f32 (n, C)
    partial_dot_ptr,     # f32 (n, C)
    invstd_ptr,          # f32 (C,)
    sum_out_ptr,         # f32 (C,)
    dot_out_ptr,         # f32 (C,)
    weight_grad_out_ptr, # f32 (C,)
    N_: ct.Constant[int],
):
    c = ct.bid(0)
    acc_sum = ct.zeros((1,), dtype=ct.float32)
    acc_dot = ct.zeros((1,), dtype=ct.float32)
    for n in range(N_):
        s = ct.load(partial_sum_ptr, index=(n, c), shape=(1, 1))
        d = ct.load(partial_dot_ptr, index=(n, c), shape=(1, 1))
        acc_sum = acc_sum + ct.reshape(s, (1,))
        acc_dot = acc_dot + ct.reshape(d, (1,))
    invstd = ct.load(invstd_ptr, index=(c,), shape=(1,))
    ct.store(sum_out_ptr, index=(c,), tile=acc_sum)
    ct.store(dot_out_ptr, index=(c,), tile=acc_dot)
    ct.store(weight_grad_out_ptr, index=(c,), tile=acc_dot * invstd)


@ct.kernel
def _bn_backward_epilogue_kernel(
    producer_ptr,
    centered_ptr,
    invstd_ptr,
    weight_ptr,
    sum_ptr,
    dot_ptr,
    out_ptr,
    HW_PAD: ct.Constant[int],
    SCALE_C: ct.Constant[float],
):
    n = ct.bid(0)
    c = ct.bid(1)
    producer = ct.load(producer_ptr, index=(n, c, 0), shape=(1, 1, HW_PAD),
                       padding_mode=ct.PaddingMode.ZERO)
    centered = ct.load(centered_ptr, index=(n, c, 0), shape=(1, 1, HW_PAD),
                       padding_mode=ct.PaddingMode.ZERO)
    invstd = ct.load(invstd_ptr, index=(c,), shape=(1,))
    weight = ct.load(weight_ptr, index=(c,), shape=(1,))
    sum_v = ct.load(sum_ptr, index=(c,), shape=(1,))
    dot_v = ct.load(dot_ptr, index=(c,), shape=(1,))

    invstd_3d = ct.reshape(invstd, (1, 1, 1))
    weight_3d = ct.reshape(weight, (1, 1, 1))
    sum_3d = ct.reshape(sum_v, (1, 1, 1))
    dot_3d = ct.reshape(dot_v, (1, 1, 1))

    mean_term = sum_3d * SCALE_C
    dot_scaled = dot_3d * SCALE_C
    variance_term = dot_scaled * (invstd_3d * invstd_3d)
    output_scale = invstd_3d * weight_3d
    after_variance = producer - centered * variance_term
    after_mean = after_variance - mean_term
    out = ct.astype(after_mean * output_scale, ct.bfloat16)
    ct.store(out_ptr, index=(n, c, 0), tile=out)


@oracle_impl(hardware="B200", point="82bc110b", BLOCK_K=1024, EPILOGUE_BLOCK=512)
def oracle_forward(inputs, *, BLOCK_K, EPILOGUE_BLOCK):
    (arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1,
     _shape0, _shape1) = inputs
    device = arg2_1.device

    n = int(arg2_1.shape[0])
    c = int(arg2_1.shape[1])
    h = int(arg2_1.shape[2])
    w = int(arg2_1.shape[3])
    hw = h * w

    # Shuffle in torch (matches the layout requirements).
    slice_1 = arg0_1[:, :116, :, :]
    cat = torch.cat([slice_1, arg1_1], dim=1)
    view = cat.view(n, 116, 2, h, w)
    permute = view.permute(0, 2, 1, 3, 4)
    clone = permute.contiguous()
    view_1 = clone.view(n, 232, h, w)
    slice_2 = view_1[:, 116:232, :, :]

    # BN forward + ReLU (torch — matches Triton's channels-last handling).
    bn = (arg2_1.to(torch.float32) - arg3_1) * arg4_1
    bn = bn * arg5_1.view(1, c, 1, 1) + arg6_1.view(1, c, 1, 1)
    bn_bf = bn.to(torch.bfloat16)
    relu = torch.relu(bn_bf)
    le = relu <= 0
    where_result = torch.where(le, arg7_1, slice_2)
    producer_f = where_result.to(torch.float32)

    squeeze = arg3_1.view(c)
    squeeze_1 = arg4_1.view(c)
    sub_1 = arg2_1.to(torch.float32) - arg3_1

    producer_contig = producer_f.contiguous()
    centered_contig = sub_1.contiguous()

    # cuTile partial-reduce + finalize instead of torch sum reductions.
    partial_sum = torch.empty((n, c), device=device, dtype=torch.float32)
    partial_dot = torch.empty((n, c), device=device, dtype=torch.float32)
    sum_1 = torch.empty((c,), device=device, dtype=torch.float32)
    sum_2 = torch.empty((c,), device=device, dtype=torch.float32)
    mul_10 = torch.empty((c,), device=device, dtype=torch.float32)

    hw_pad = 1 << (hw - 1).bit_length()
    stream = torch.cuda.current_stream()

    ct.launch(
        stream,
        (n, c, 1),
        _partial_reduce_kernel,
        (producer_contig.view(n, c, hw),
         centered_contig.view(n, c, hw),
         partial_sum, partial_dot, hw_pad),
    )
    ct.launch(
        stream,
        (c, 1, 1),
        _finalize_kernel,
        (partial_sum, partial_dot, squeeze_1, sum_1, sum_2, mul_10, n),
    )

    out_contig = torch.empty((n, c, h, w), device=device, dtype=torch.bfloat16)
    ct.launch(
        stream,
        (n, c, 1),
        _bn_backward_epilogue_kernel,
        (producer_contig.view(n, c, hw),
         centered_contig.view(n, c, hw),
         squeeze_1,   # invstd
         arg5_1,      # weight
         sum_1,
         sum_2,
         out_contig.view(n, c, hw),
         hw_pad, SCALE),
    )
    return view_1, sum_1, mul_10, out_contig
