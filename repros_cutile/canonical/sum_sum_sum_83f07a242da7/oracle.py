"""cuTile port of sum_sum_sum_83f07a242da7: grouped BN-backward + ReLU gate.

Mirrors Triton's structure: one kernel per (batch, group) pair that fuses
BN forward-gate, per-channel spatial reductions, dependent grouped
reductions, full bf16 epilogue, and atomic-add into two f32 vector
accumulators. A tiny prologue zeros the accumulators.

Grid: (BATCH * GROUPS,) = 128 * 32 = 4096 programs. 32 fixed groups.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


GROUPS = 32
GROUP_SCALE = 0.0078125


@ct.kernel
def _zero_vectors_kernel(
    out_vec_ptr,
    out_sum_ptr,
    BLOCK_C: ct.Constant[int],
):
    zeros = ct.zeros((BLOCK_C,), dtype=ct.float32)
    ct.store(out_vec_ptr, index=(0,), tile=zeros)
    ct.store(out_sum_ptr, index=(0,), tile=zeros)


@ct.kernel
def _grouped_bn_backward_kernel(
    arg0_ptr,        # bf16, flat 1D view over shape (BATCH,C,H,W) strided
    arg1_ptr,        # bf16, flat 1D view over shape (BATCH,C,H,W) strided
    arg2_ptr,        # f32  [BATCH * GROUPS] mean (flat)
    arg3_ptr,        # f32  [BATCH * GROUPS] invstd (flat)
    arg4_ptr,        # f32  [C] gamma
    arg5_ptr,        # f32  [C] beta
    arg6_ptr,        # f32  [1] scalar
    out_vec_ptr,     # f32  [C]
    out_sum_ptr,     # f32  [C]
    out_full_ptr,    # bf16, flat 1D view over shape (BATCH,C,H,W)
    GROUPS_C: ct.Constant[int],
    GROUP_WIDTH: ct.Constant[int],
    HW_SIZE: ct.Constant[int],
    ARG0_S0: ct.Constant[int],
    ARG0_S1: ct.Constant[int],
    ARG0_S3: ct.Constant[int],
    ARG1_S0: ct.Constant[int],
    ARG1_S1: ct.Constant[int],
    ARG1_S3: ct.Constant[int],
    OUT_S0: ct.Constant[int],
    OUT_S1: ct.Constant[int],
    OUT_S3: ct.Constant[int],
):
    pid = ct.bid(0)
    n = pid // GROUPS_C
    group = pid - n * GROUPS_C

    r = ct.arange(GROUP_WIDTH, dtype=ct.int32)
    hw = ct.arange(HW_SIZE, dtype=ct.int32)
    channel = group * GROUP_WIDTH + r  # (GROUP_WIDTH,) int32

    # Build offsets for arg0/arg1: (GROUP_WIDTH, HW_SIZE) shape.
    ch_2d = ct.reshape(channel, (GROUP_WIDTH, 1))
    hw_2d = ct.reshape(hw, (1, HW_SIZE))

    input0_offsets = n * ARG0_S0 + ch_2d * ARG0_S1 + hw_2d * ARG0_S3
    input1_offsets = n * ARG1_S0 + ch_2d * ARG1_S1 + hw_2d * ARG1_S3
    output_offsets = n * OUT_S0 + ch_2d * OUT_S1 + hw_2d * OUT_S3

    x0_bf = ct.gather(arg0_ptr, input0_offsets)
    x1_bf = ct.gather(arg1_ptr, input1_offsets)
    x0 = ct.astype(x0_bf, ct.float32)
    x1 = ct.astype(x1_bf, ct.float32)

    # Scalars: mean/invstd indexed by n*GROUPS+group; gamma/beta by channel.
    ng = n * GROUPS_C + group
    mean_tile = ct.load(arg2_ptr, index=(ng,), shape=(1,))
    invstd_tile = ct.load(arg3_ptr, index=(ng,), shape=(1,))
    gamma_vec = ct.gather(arg4_ptr, channel)
    beta_vec = ct.gather(arg5_ptr, channel)
    scalar_v = ct.load(arg6_ptr, index=(0,), shape=(1,))

    mean_scalar = ct.reshape(mean_tile, (1, 1))
    invstd_scalar = ct.reshape(invstd_tile, (1, 1))
    gamma_2d = ct.reshape(gamma_vec, (GROUP_WIDTH, 1))
    beta_2d = ct.reshape(beta_vec, (GROUP_WIDTH, 1))
    scalar_2d = ct.reshape(scalar_v, (1, 1))

    normalized = (x1 - mean_scalar) * invstd_scalar
    relu_input = normalized * gamma_2d + beta_2d
    scalar_bcast = ct.broadcast_to(scalar_2d, (GROUP_WIDTH, HW_SIZE))
    where_value = ct.where(relu_input <= 0.0, scalar_bcast, x0)

    # Per-row sums over HW dim (axis=1) → (GROUP_WIDTH,)
    sum_mul = ct.sum(where_value * x1, axis=1)
    sum_where = ct.sum(where_value, axis=1)

    # Grouped sums (over GROUP_WIDTH channels) → scalar
    grouped_mul_v = ct.sum(sum_mul * gamma_vec)
    grouped_where_v = ct.sum(sum_where * gamma_vec)
    grouped_mul = ct.reshape(grouped_mul_v, (1,))
    grouped_where = ct.reshape(grouped_where_v, (1,))
    mean_s = ct.reshape(mean_tile, (1,))
    invstd_s = ct.reshape(invstd_tile, (1,))

    grouped_scale = (grouped_where * mean_s - grouped_mul) * invstd_s * invstd_s * invstd_s
    grouped_scale = grouped_scale * GROUP_SCALE
    grouped_bias = -(grouped_scale * mean_s) - grouped_where * invstd_s * GROUP_SCALE

    grouped_scale_bc = ct.reshape(grouped_scale, (1, 1))
    grouped_bias_bc = ct.reshape(grouped_bias, (1, 1))
    invstd_gamma = ct.reshape(invstd_s * gamma_vec, (GROUP_WIDTH, 1))
    out_full = where_value * invstd_gamma + x1 * grouped_scale_bc + grouped_bias_bc
    ct.scatter(out_full_ptr, output_offsets, ct.astype(out_full, ct.bfloat16))

    # Vector accumulator contributions (atomic-add).
    out_vec_contrib = (sum_mul - sum_where * mean_s) * invstd_s
    ct.atomic_add(out_vec_ptr, (channel,), out_vec_contrib)
    ct.atomic_add(out_sum_ptr, (channel,), sum_where)


def _next_power_of_2(value):
    return 1 << (int(value) - 1).bit_length()


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= int(dim)
    return tuple(reversed(stride))


def _run(inputs, *, group_width: int, hw_size: int, width: int):
    (
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1,
        _s0, _s1, _s2, _s3, _s4, _s5, _s6, _s7, full_shape_arg,
        _s9, _s10, vec_shape_arg,
    ) = inputs
    batch = int(arg0_1.shape[0])
    channels = int(arg0_1.shape[1])
    height = int(arg0_1.shape[2])
    _width = int(arg0_1.shape[3])
    full_shape = tuple(int(d) for d in full_shape_arg)
    vec_shape = tuple(int(d) for d in vec_shape_arg)

    device = arg0_1.device
    out_vec = torch.empty_strided(vec_shape, (1,), device=device, dtype=torch.float32)
    out_sum = torch.empty_strided(vec_shape, (1,), device=device, dtype=torch.float32)
    out_full = torch.empty_strided(
        full_shape, _contiguous_stride(full_shape),
        device=device, dtype=torch.bfloat16,
    )

    # Views for cuTile passes.
    # arg2/arg3: [BATCH, GROUPS, 1, 1] — flatten to [BATCH * GROUPS].
    arg2_flat = arg2_1.reshape(-1)
    arg3_flat = arg3_1.reshape(-1)
    arg4_flat = arg4_1.view(channels)
    arg5_flat = arg5_1.view(channels)
    arg6_flat = arg6_1.view(1)
    out_vec_view = out_vec.view(channels)
    out_sum_view = out_sum.view(channels)

    # For gather to work with strided tensors, we pass the tensor and rely on
    # cuTile's Array respecting strides via the underlying storage.
    # We use flat 1D views for gather/scatter; offsets are into the *storage*.
    # For strided arg0/arg1, offsets encode: n*S0 + c*S1 + h*S2 + w*S3.
    # We combine h and w into hw using stride S3 for the flattened index,
    # only valid when S2 == W * S3 (channels-last layouts have S2=W_stride*W).
    # For (H,W)=(1,1) that's trivial.
    # We view arg0/arg1 as 1D over the underlying storage.
    def _flat_storage(t):
        # Interpret the tensor as a 1D array of length equal to the
        # linear extent that offsets need to reach.
        n0, s0 = int(t.shape[0]), int(t.stride(0))
        # Total storage length = (n0 - 1) * s0 + max_reach
        # Simplest: view as flat over max(offset)+1.
        max_off = (n0 - 1) * s0
        for d in range(1, t.dim()):
            max_off += (int(t.shape[d]) - 1) * int(t.stride(d))
        return t.reshape(t.numel()) if t.is_contiguous() else t.as_strided((max_off + 1,), (1,))

    # arg0/arg1 have non-standard strides; flatten conceptually via as_strided.
    # But cuTile's gather takes flat indices into the array's tile-space, which
    # is derived from the underlying tensor. Rather than trying to flatten,
    # we index via a flat storage view.
    arg0_storage = arg0_1.reshape(-1) if arg0_1.is_contiguous() else \
        torch.as_strided(arg0_1, (arg0_1.numel(),), (1,))
    arg1_storage = arg1_1.reshape(-1) if arg1_1.is_contiguous() else \
        torch.as_strided(arg1_1, (arg1_1.numel(),), (1,))
    out_storage = out_full.reshape(-1) if out_full.is_contiguous() else \
        torch.as_strided(out_full, (out_full.numel(),), (1,))

    stream = torch.cuda.current_stream()

    zero_block = _next_power_of_2(channels)
    ct.launch(
        stream, (1, 1, 1), _zero_vectors_kernel,
        (out_vec_view, out_sum_view, zero_block),
    )

    # Kernel launch. Note: since H*W = hw_size and we flatten (h,w) into hw,
    # the stride to use for hw is arg.stride(3) — assuming stride(2) = W * stride(3).
    ct.launch(
        stream, (batch * GROUPS, 1, 1), _grouped_bn_backward_kernel,
        (
            arg0_storage, arg1_storage, arg2_flat, arg3_flat,
            arg4_flat, arg5_flat, arg6_flat,
            out_vec_view, out_sum_view, out_storage,
            GROUPS, group_width, hw_size,
            int(arg0_1.stride(0)), int(arg0_1.stride(1)), int(arg0_1.stride(3)),
            int(arg1_1.stride(0)), int(arg1_1.stride(1)), int(arg1_1.stride(3)),
            int(out_full.stride(0)), int(out_full.stride(1)), int(out_full.stride(3)),
        ),
    )
    return out_vec, out_sum, out_full


# e311375b: (T([128,64,8,8], bf16, channels-last-like), ...)
@oracle_impl(hardware="B200", point="e311375b", group_width=2, hw_size=64, width=8)
# c9ab4bb9: (T([128,128,4,4], bf16, channels-last-like), ...)
@oracle_impl(hardware="B200", point="c9ab4bb9", group_width=4, hw_size=16, width=4)
# eeb75e7f: (T([128,256,2,2], bf16, channels-last-like), ...)
@oracle_impl(hardware="B200", point="eeb75e7f", group_width=8, hw_size=4, width=2)
# 13564432: (T([128,512,1,1], bf16), ...)
@oracle_impl(hardware="B200", point="13564432", group_width=16, hw_size=1, width=1)
def oracle_forward(inputs, *, group_width: int, hw_size: int, width: int):
    return _run(inputs, group_width=group_width, hw_size=hw_size, width=width)
