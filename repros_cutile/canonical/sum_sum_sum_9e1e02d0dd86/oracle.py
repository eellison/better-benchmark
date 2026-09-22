"""cuTile port of sum_sum_sum_9e1e02d0dd86: DenseNet 121 dual BN-backward + maxpool bwd.

Three cuTile kernels mirroring Triton's structure:
- ``_slice_add6_kernel`` — multi-arg slice-add chain (arg0..arg5[:, 0:64]).
- ``_channel_sum_dot_kernel`` — per-channel ``sum(where)`` and
  ``sum(where * centered)``. Launched twice, once per BN-backward branch,
  matching Triton's two partial-reduction kernels.

Everything else (BN backward epilogue, maxpool backward scatter, second BN
gate) still uses torch, which is fine for structure-fairness.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _slice_add6_kernel(
    s0_ptr,     # bf16 [N]
    s1_ptr,
    s2_ptr,
    s3_ptr,
    s4_ptr,
    s5_ptr,
    out_ptr,    # bf16 [N]
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    a0 = ct.load(s0_ptr, index=(pid,), shape=(BLOCK,))
    a1 = ct.load(s1_ptr, index=(pid,), shape=(BLOCK,))
    a2 = ct.load(s2_ptr, index=(pid,), shape=(BLOCK,))
    a3 = ct.load(s3_ptr, index=(pid,), shape=(BLOCK,))
    a4 = ct.load(s4_ptr, index=(pid,), shape=(BLOCK,))
    a5 = ct.load(s5_ptr, index=(pid,), shape=(BLOCK,))
    # Bf16-rounded pairwise adds
    def add_bf(x, y):
        return ct.astype(
            ct.astype(x, ct.float32) + ct.astype(y, ct.float32),
            ct.bfloat16,
        )
    t = add_bf(a0, a1)
    t = add_bf(t, a2)
    t = add_bf(t, a3)
    t = add_bf(t, a4)
    t = add_bf(t, a5)
    ct.store(out_ptr, index=(pid,), tile=t)


@ct.kernel
def _channel_sum_dot_kernel(
    where_ptr,       # f32 [C, K_TOTAL]
    centered_ptr,    # f32 [C, K_TOTAL]
    sum_out_ptr,     # f32 [C]
    dot_out_ptr,     # f32 [C]
    K_TOTAL_: ct.Constant[int],
    BLOCK_K: ct.Constant[int],
    N_K_BLOCKS: ct.Constant[int],
):
    c = ct.bid(0)
    sum_acc = ct.zeros((), dtype=ct.float32)
    dot_acc = ct.zeros((), dtype=ct.float32)
    for kb in range(N_K_BLOCKS):
        where_f = ct.load(
            where_ptr, index=(c, kb), shape=(1, BLOCK_K),
            padding_mode=ct.PaddingMode.ZERO,
        )
        centered_f = ct.load(
            centered_ptr, index=(c, kb), shape=(1, BLOCK_K),
            padding_mode=ct.PaddingMode.ZERO,
        )
        ks = ct.arange(BLOCK_K, dtype=ct.int32) + kb * BLOCK_K
        valid = ks < K_TOTAL_
        valid_2d = ct.reshape(valid, (1, BLOCK_K))
        zero_f = ct.full((1, BLOCK_K), 0.0, dtype=ct.float32)
        where_masked = ct.where(valid_2d, where_f, zero_f)
        centered_masked = ct.where(valid_2d, centered_f, zero_f)
        sum_acc = sum_acc + ct.sum(where_masked)
        dot_acc = dot_acc + ct.sum(where_masked * centered_masked)
    ct.store(sum_out_ptr, index=(c,), tile=ct.reshape(sum_acc, (1,)))
    ct.store(dot_out_ptr, index=(c,), tile=ct.reshape(dot_acc, (1,)))


BLOCK_K = 4096


def _launch_channel_sum_dot(stream, where_f, centered_f, C):
    """Run per-channel sum & dot via cuTile.

    ``where_f`` and ``centered_f`` are f32 tensors of shape ``[N, C, H, W]``.
    Returns (sum_where, sum_where_times_centered), each shape ``[C]`` f32.
    """
    N, C_, H, W = where_f.shape
    assert C_ == C
    K = int(N) * int(H) * int(W)
    where_ck = where_f.permute(1, 0, 2, 3).contiguous().view(C, K)
    centered_ck = centered_f.permute(1, 0, 2, 3).contiguous().view(C, K)
    sum_out = torch.empty((C,), device=where_f.device, dtype=torch.float32)
    dot_out = torch.empty((C,), device=where_f.device, dtype=torch.float32)
    n_k_blocks = (K + BLOCK_K - 1) // BLOCK_K
    ct.launch(
        stream, (C, 1, 1), _channel_sum_dot_kernel,
        (where_ck, centered_ck, sum_out, dot_out,
         K, BLOCK_K, n_k_blocks),
    )
    return sum_out, dot_out


@oracle_impl(hardware="B200", point="0b922608")
def oracle_forward(inputs):
    (arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1,
     arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1,
     arg15_1, arg16_1, arg17_1, arg18_1,
     _sh0, _sh1, _sh2, _sh3, _sh4, _sh5, _sh6) = inputs
    device = arg0_1.device

    # ---- cuTile kernel #1: slice-add6 chain ----
    slices = [t[:, 0:64].contiguous() for t in
              (arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1)]
    N = slices[0].numel()
    BLOCK = 1024
    total_pad = ((N + BLOCK - 1) // BLOCK) * BLOCK
    pads = []
    for s in slices:
        p = torch.zeros((total_pad,), device=device, dtype=torch.bfloat16)
        p[:N].copy_(s.view(-1))
        pads.append(p)
    out_pad = torch.empty_like(pads[0])
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (total_pad // BLOCK, 1, 1),
        _slice_add6_kernel,
        (*pads, out_pad, BLOCK),
    )
    add_4 = out_pad[:N].view_as(slices[0])

    # ---- ReLU-gate + BN backward branch 1 ----
    B, C, H, W = arg6_1.shape
    B, C, H, W = int(B), int(C), int(H), int(W)
    le = arg6_1 <= 0
    where = torch.where(le, arg7_1, arg8_1)
    cet = where.float()
    sub = arg9_1.float() - arg10_1
    # sub broadcasts to [B, C, H, W] (arg10_1 is [1, C, 1, 1]); make it dense.
    sub_bcast = sub.expand(B, C, H, W).contiguous()

    # cuTile kernel #2: per-channel sum(cet) and sum(cet * sub)
    sum_1, sum_2 = _launch_channel_sum_dot(stream, cet, sub_bcast, C)

    L1 = 7.971938775510203e-05
    mul_1 = sum_1 * L1
    unsq_2 = mul_1.view(1, C, 1, 1)
    mul_2 = sum_2 * L1
    mul_3 = arg11_1 * arg11_1
    mul_4 = mul_2 * mul_3
    unsq_5 = mul_4.view(1, C, 1, 1)
    mul_5 = arg11_1 * arg12_1
    unsq_8 = mul_5.view(1, C, 1, 1)
    mul_6 = sub * unsq_5
    sub_1 = cet - mul_6
    sub_2 = sub_1 - unsq_2
    mul_7 = sub_2 * unsq_8
    mul_8 = sum_2 * arg11_1
    cet_2 = mul_7.to(torch.bfloat16)
    add_5 = add_4 + cet_2

    # ---- Maxpool backward via scatter (torch) ----
    add_5_flat = add_5.view(256, 3136).float()
    indices = torch.ops.prims._low_memory_max_pool_offsets_to_indices.default(
        arg13_1, [3, 3], [112, 112], [2, 2], [1, 1], [1, 1],
    )
    ind_2d = indices.view(256, 3136)
    scatter = torch.zeros((256, 12544), device=device, dtype=torch.float32).scatter_add_(1, ind_2d, add_5_flat)
    view_2 = scatter.view(4, 64, 112, 112)
    cet_4 = view_2.to(torch.bfloat16)

    # ---- Second BN branch (ReLU-gate against affine of arg14_1) ----
    sub_3 = arg14_1.float() - arg15_1
    mul_9 = sub_3 * arg16_1
    unsq_10 = arg17_1.view(1, C, 1, 1)
    mul_10 = mul_9 * unsq_10
    unsq_12 = arg18_1.view(1, C, 1, 1)
    add_6 = mul_10 + unsq_12
    cet_5 = add_6.to(torch.bfloat16)
    relu_out = torch.relu(cet_5)
    le_1 = relu_out <= 0
    where_1 = torch.where(le_1, arg7_1, cet_4)
    cet_6 = where_1.float()
    squeeze = arg15_1.view(C)
    unsq_15 = squeeze.view(1, C, 1, 1)
    sub_4 = arg14_1.float() - unsq_15
    _, _, H2, W2 = arg14_1.shape
    H2, W2 = int(H2), int(W2)
    sub4_bcast = sub_4.expand(B, C, H2, W2).contiguous()

    # cuTile kernel #3: per-channel sum(cet_6) and sum(cet_6 * sub_4)
    sum_3, sum_4 = _launch_channel_sum_dot(stream, cet_6, sub4_bcast, C)

    L2 = 1.992984693877551e-05
    mul_12 = sum_3 * L2
    unsq_18 = mul_12.view(1, C, 1, 1)
    mul_13 = sum_4 * L2
    squeeze_1 = arg16_1.view(C)
    mul_14 = squeeze_1 * squeeze_1
    mul_15 = mul_13 * mul_14
    unsq_21 = mul_15.view(1, C, 1, 1)
    mul_16 = squeeze_1 * arg17_1
    unsq_24 = mul_16.view(1, C, 1, 1)
    mul_17 = sub_4 * unsq_21
    sub_5 = cet_6 - mul_17
    sub_6 = sub_5 - unsq_18
    mul_18 = sub_6 * unsq_24
    mul_19 = sum_4 * squeeze_1
    cet_8 = mul_18.to(torch.bfloat16)
    return sum_1, mul_8, sum_3, mul_19, cet_8
