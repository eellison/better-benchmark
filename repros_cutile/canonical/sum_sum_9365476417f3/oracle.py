"""cuTile port of sum_sum_9365476417f3: ResNeXt maxpool-backward scatter
+ BN forward + BN backward with in-kernel reductions.

Mirrors Triton's 3-kernel plan:
1. `_partials_kernel` (grid = (num_groups, C)): compute per-channel partial
   sum(producer) and sum(producer * sub) via in-kernel `ct.sum`.
2. `_finalize_kernel` (grid = (C_blocks,)): reduce partials over groups,
   store `sum1`, `sum2`, and `mul10 = sum2 * invstd`.
3. `_bn_backward_epilogue_kernel` (grid = (N, C)): BN-backward elementwise
   epilogue using the finalized sums.

Torch handles the scatter and BN forward (feeds `producer_f`, `sub_f`);
cuTile handles both channel reductions in-kernel and the BN-backward.
GROUP_R=1024 matches Triton; FINAL_BLOCK_C=16 for the finalize kernel.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


SCALE = 9.964923469387754e-06


def _next_power_of_2(v):
    return 1 << (int(v) - 1).bit_length()


@ct.kernel
def _partials_kernel(
    producer_ptr,        # f32 flat [N*C*HW]
    sub_ptr,             # f32 flat [N*C*HW]
    partial_sum1_ptr,    # f32 [num_groups * C]
    partial_sum2_ptr,    # f32 [num_groups * C]
    R_: ct.Constant[int],
    HW_: ct.Constant[int],
    C_: ct.Constant[int],
    GROUP_R_: ct.Constant[int],
):
    group = ct.bid(0)
    c = ct.bid(1)
    lanes = ct.arange(GROUP_R_, dtype=ct.int32)
    r_offsets = group * GROUP_R_ + lanes
    active = r_offsets < R_
    n = r_offsets // HW_
    spatial = r_offsets - n * HW_
    flat_offsets = n * (C_ * HW_) + c * HW_ + spatial

    prod = ct.gather(producer_ptr, flat_offsets)
    sub = ct.gather(sub_ptr, flat_offsets)
    prod = ct.where(active, prod, 0.0)
    sub = ct.where(active, sub, 0.0)

    sum_prod = ct.sum(prod)  # scalar
    sum_dot = ct.sum(prod * sub)  # scalar

    ct.store(partial_sum1_ptr, index=(group * C_ + c,),
             tile=ct.reshape(sum_prod, (1,)))
    ct.store(partial_sum2_ptr, index=(group * C_ + c,),
             tile=ct.reshape(sum_dot, (1,)))


@ct.kernel
def _finalize_kernel(
    partial_sum1_ptr,   # f32 [num_groups * C]
    partial_sum2_ptr,   # f32 [num_groups * C]
    invstd_ptr,         # f32 [C]
    sum1_ptr,           # f32 [C]
    sum2_ptr,           # f32 [C]
    mul10_ptr,          # f32 [C]
    NUM_GROUPS_: ct.Constant[int],
    C_: ct.Constant[int],
    BLOCK_GROUPS_: ct.Constant[int],
):
    c = ct.bid(0)
    groups = ct.arange(BLOCK_GROUPS_, dtype=ct.int32)
    active = groups < NUM_GROUPS_
    offsets = groups * C_ + c

    p1 = ct.gather(partial_sum1_ptr, offsets)
    p2 = ct.gather(partial_sum2_ptr, offsets)
    p1 = ct.where(active, p1, 0.0)
    p2 = ct.where(active, p2, 0.0)
    s1 = ct.sum(p1)
    s2 = ct.sum(p2)

    invstd_scalar = ct.gather(invstd_ptr, ct.broadcast_to(c, (1,)))
    invstd = ct.reshape(invstd_scalar, (1,))

    ct.store(sum1_ptr, index=(c,), tile=ct.reshape(s1, (1,)))
    ct.store(sum2_ptr, index=(c,), tile=ct.reshape(s2, (1,)))
    mul10_val = s2 * ct.reshape(invstd, ())
    ct.store(mul10_ptr, index=(c,), tile=ct.reshape(mul10_val, (1,)))


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


@oracle_impl(hardware="B200", point="c3354c52", GROUP_R=1024, REDUCE_BLOCK_R=64, BLOCK_C=16, FINAL_BLOCK_C=16, OUT_BLOCK_R=128)
def oracle_forward(inputs, *, GROUP_R, REDUCE_BLOCK_R, BLOCK_C, FINAL_BLOCK_C, OUT_BLOCK_R):
    (arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8,
     s0, s1, s2, s3, s4, s5, s6) = inputs
    device = arg0.device
    n = int(arg0.shape[0])
    c = int(arg0.shape[1])
    out_h = int(arg3.shape[2])
    out_w = int(arg3.shape[3])
    hw_out = out_h * out_w
    r = n * hw_out

    # Producer: scatter + BN + ReLU-gate + where. Kept in torch since cuTile
    # can't easily do fixed-window scatter; ct.sum reductions below match
    # Triton's kernel-side reduction.
    add_bf = arg0 + arg1
    kernel_size = list(s2)
    input_size = list(s3)
    stride = list(s4)
    indices = torch.ops.prims._low_memory_max_pool_offsets_to_indices.default(
        arg2, kernel_size, input_size, stride, [1, 1], [1, 1]
    )
    view_add = add_bf.view(-1, 3136)
    view_idx = indices.view(-1, 3136)
    zeros = torch.zeros((view_add.shape[0], 12544), dtype=torch.float32, device=device)
    scatter_out = zeros.scatter_add_(1, view_idx, view_add.to(torch.float32))
    scatter_bf = scatter_out.to(torch.bfloat16).view(n, c, 112, 112)

    bn = (arg3.to(torch.float32) - arg4) * arg5
    bn = bn * arg6.view(1, c, 1, 1) + arg7.view(1, c, 1, 1)
    bn_bf = bn.to(torch.bfloat16)
    relu = torch.relu(bn_bf)
    le = relu <= 0
    where_result = torch.where(le, arg8, scatter_bf)
    producer_f = where_result.to(torch.float32).contiguous()
    squeeze_5 = arg5.view(c)
    sub_f = (arg3.to(torch.float32) - arg4).contiguous()

    # 3-kernel cuTile plan.
    num_groups = (r + GROUP_R - 1) // GROUP_R
    partial_sum1 = torch.empty((num_groups, c), device=device, dtype=torch.float32)
    partial_sum2 = torch.empty((num_groups, c), device=device, dtype=torch.float32)
    sum1 = torch.empty((c,), device=device, dtype=torch.float32)
    sum2 = torch.empty((c,), device=device, dtype=torch.float32)
    mul10 = torch.empty((c,), device=device, dtype=torch.float32)

    out_contig = torch.empty((n, c, out_h, out_w), device=device, dtype=torch.bfloat16)
    hw_pad = 1 << (hw_out - 1).bit_length()

    def _flat(t):
        size = 1
        for s in t.shape:
            size *= s
        return t.contiguous().view(size)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (num_groups, c, 1),
        _partials_kernel,
        (_flat(producer_f), _flat(sub_f),
         partial_sum1.view(-1), partial_sum2.view(-1),
         r, hw_out, c, GROUP_R),
    )
    ct.launch(
        stream,
        (c, 1, 1),
        _finalize_kernel,
        (partial_sum1.view(-1), partial_sum2.view(-1),
         squeeze_5,
         sum1, sum2, mul10,
         num_groups, c, _next_power_of_2(num_groups)),
    )
    ct.launch(
        stream,
        (n, c, 1),
        _bn_backward_epilogue_kernel,
        (producer_f.view(n, c, hw_out),
         sub_f.view(n, c, hw_out),
         squeeze_5,   # invstd
         arg6,        # weight
         sum1,
         sum2,
         out_contig.view(n, c, hw_out),
         hw_pad, SCALE),
    )
    return sum1, mul10, out_contig
