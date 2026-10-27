"""cuTile port of var_mean_var_mean_4cdcce28fdf4: dual GroupNorm + relu + bf16 cast.

Ports the Triton port to cuTile. The reference computes two independent group
norms (128, 32 groups) over pow2 elements per group, applies per-channel
weight/bias, sums both branches, does relu, and rounds bf16 to bf16. All
reduction dimensions are pow2 so we tile per-group.

Simplest port: run the group-mean/var reduction and normalization in cuTile
(one kernel per group covers a (BATCH*GROUP, ELEMS) tile). Compute final
affine/add/relu/bf16 with cuTile pointwise flat over the (128, C) space
using ct.gather for channel-indexed weights/biases.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 128
GROUPS = 32


@ct.kernel
def _group_mean_var_kernel(
    x_ptr,           # f32[BATCH, GROUPS, ELEMS]
    mean_ptr,        # f32[BATCH, GROUPS]
    invstd_ptr,      # f32[BATCH, GROUPS]
    ELEMS: ct.Constant[int],
    INV_ELEMS: ct.Constant[float],
):
    row = ct.bid(0)  # global row over BATCH*GROUPS
    x = ct.load(x_ptr, index=(row, 0), shape=(1, ELEMS))
    mean = ct.sum(x, axis=1) * INV_ELEMS               # shape (1,)
    centered = x - ct.reshape(mean, (1, 1))            # shape (1, ELEMS)
    var = ct.sum(centered * centered, axis=1) * INV_ELEMS
    invstd = 1.0 / ct.sqrt(var + 1.0e-5)               # shape (1,)
    ct.store(mean_ptr, index=(row,), tile=mean)
    ct.store(invstd_ptr, index=(row,), tile=invstd)


@ct.kernel
def _dual_gn_epilogue_kernel(
    x0_ptr,       # f32[NUMEL]
    x1_ptr,       # f32[NUMEL]
    mean0_ptr,    # f32[BATCH*GROUPS]
    invstd0_ptr,  # f32[BATCH*GROUPS]
    mean1_ptr,    # f32[BATCH*GROUPS]
    invstd1_ptr,  # f32[BATCH*GROUPS]
    w0_ptr,       # f32[C]
    b0_ptr,       # f32[C]
    w1_ptr,       # f32[C]
    b1_ptr,       # f32[C]
    relu_out_ptr, # f32[NUMEL]
    bf16_out_ptr, # bf16[NUMEL]
    C_: ct.Constant[int],
    ELEMS_HW: ct.Constant[int],   # ELEMS * (1 if HW==1 else HW factor)
    ELEMS_PER_GROUP: ct.Constant[int],
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    idx = pid * BLOCK + ct.arange(BLOCK, dtype=ct.int32)
    # NUMEL = BATCH * C * HW; but we're operating on the (BATCH*C*HW) flat.
    # Original view for GN: (BATCH, GROUPS, ELEMS_PER_GROUP * HW=ELEMS).
    # Then after view back to (BATCH, C, H, W): elements are laid out as
    # (batch, group, elem_in_group, hw)... but since view is same storage
    # and reshape is arbitrary interleaving, we need to derive the (batch,
    # group, hw) index from flat.
    # The reference (128, 512, 1, 1) -> view (128, 32, 16, 1) preserves storage.
    # For flat idx into (BATCH, C, H, W) contiguous: batch = idx // (C*HW),
    # then rem = idx % (C*HW); channel = rem // HW; hw_idx = rem % HW.
    # Now for GN view (BATCH, GROUPS, ELEMS_PER_GROUP, HW) where
    # ELEMS_PER_GROUP*HW = ELEMS_HW: group = channel // (C/GROUPS),
    # elem_in_group = channel % (C/GROUPS), then GN reduces over
    # (ELEMS_PER_GROUP, HW) = the pair whose flat linear is
    # elem_in_group * HW + hw_idx.
    # The GN mean/invstd is per (batch, group), i.e. per row = batch*GROUPS + group.
    # ChannelsPerGroup = C_ // GROUPS.
    channels_per_group = C_ // GROUPS

    # For flat idx (into contiguous (BATCH, C, H, W)):
    # HW is small so we compute like this.
    # But we don't have HW here — need to pass. Instead pass ELEMS_HW = elems_per_group*HW.
    # C_ * HW = ELEMS_HW * GROUPS (since GROUPS*ELEMS_HW = C*HW).
    # Actually total_per_batch = C_ * HW. So batch = idx // (C_*HW).
    # For simplicity, pass HW separately. We inferred elems_per_group = C_ // GROUPS.
    total_per_batch = C_ * (ELEMS_HW // channels_per_group)  # = C_ * HW
    batch = idx // total_per_batch
    rem = idx - batch * total_per_batch
    hw = ELEMS_HW // channels_per_group
    channel = rem // hw
    group = channel // channels_per_group
    row = batch * GROUPS + group

    x0 = ct.load(x0_ptr, index=(pid,), shape=(BLOCK,))
    x1 = ct.load(x1_ptr, index=(pid,), shape=(BLOCK,))
    mean0 = ct.gather(mean0_ptr, row)
    invstd0 = ct.gather(invstd0_ptr, row)
    mean1 = ct.gather(mean1_ptr, row)
    invstd1 = ct.gather(invstd1_ptr, row)
    w0 = ct.gather(w0_ptr, channel)
    b0 = ct.gather(b0_ptr, channel)
    w1 = ct.gather(w1_ptr, channel)
    b1 = ct.gather(b1_ptr, channel)

    normalized0 = (x0 - mean0) * invstd0
    normalized1 = (x1 - mean1) * invstd1
    add0 = normalized0 * w0 + b0
    add1 = normalized1 * w1 + b1
    total = add0 + add1
    zero = ct.astype(ct.zeros((BLOCK,), dtype=ct.float32), ct.float32)
    relu = ct.where(total > zero, total, zero)
    ct.store(relu_out_ptr, index=(pid,), tile=relu)
    ct.store(bf16_out_ptr, index=(pid,), tile=ct.astype(relu, ct.bfloat16))


def _run(inputs, *, CHANNELS, HEIGHT, WIDTH, GROUP_ELEMS):
    (arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1,
     _shape0, _shape1, _shape2, _shape3) = inputs

    x0 = arg0_1.to(torch.float32)  # (128, C, 1, 1)
    x1 = arg3_1.to(torch.float32)
    HW = HEIGHT * WIDTH
    ELEMS = GROUP_ELEMS * HW  # elements per group per batch = C/GROUPS * HW
    # For d227844f: C=512, GROUPS=32, HW=1, ELEMS_PER_GROUP=16 (GROUP_ELEMS=16). GROUP_ELEMS is ELEMS from view.
    # Actually reviewing: view [128, 32, GROUP_ELEMS, ?] where the last two dims are the reduction dims.
    # The reduction is over [2, 3] so total reduction size = GROUP_ELEMS (view dim 2) * view dim 3.
    # For d227844f the view is [128, 32, 16, 1] so dim2=16, dim3=1. Reduction elems = 16.
    # For 1fa56db2 the view is [128, 32, 8, 4] so dim2=8, dim3=4. Reduction elems = 32.
    # For ade33c47 the view is [128, 32, 4, 16] so dim2=4, dim3=16. Reduction elems = 64.
    # In all cases the total reduced count = C/GROUPS * HW = channels_per_group * HW.
    # And GROUP_ELEMS (the ELEMS constant we accept) = C/GROUPS.
    reduction_total = (CHANNELS // GROUPS) * HW

    # Reshape to (BATCH, GROUPS, reduction_total) for the per-group reduction.
    x0_view = x0.reshape(BATCH, GROUPS, reduction_total)
    x1_view = x1.reshape(BATCH, GROUPS, reduction_total)

    mean0 = torch.empty((BATCH * GROUPS,), device=x0.device, dtype=torch.float32)
    invstd0 = torch.empty((BATCH * GROUPS,), device=x0.device, dtype=torch.float32)
    mean1 = torch.empty((BATCH * GROUPS,), device=x0.device, dtype=torch.float32)
    invstd1 = torch.empty((BATCH * GROUPS,), device=x0.device, dtype=torch.float32)

    # x0_view is fresh from x0.reshape; reshape again is metadata-only when
    # backing storage is contiguous. No extra .contiguous() needed.
    x0_view_flat = x0_view.reshape(BATCH * GROUPS, reduction_total)
    x1_view_flat = x1_view.reshape(BATCH * GROUPS, reduction_total)

    stream = torch.cuda.current_stream()
    inv_elems = 1.0 / reduction_total
    ct.launch(
        stream, (BATCH * GROUPS, 1, 1),
        _group_mean_var_kernel,
        (x0_view_flat, mean0, invstd0, reduction_total, inv_elems),
    )
    ct.launch(
        stream, (BATCH * GROUPS, 1, 1),
        _group_mean_var_kernel,
        (x1_view_flat, mean1, invstd1, reduction_total, inv_elems),
    )

    # weight/bias args are already contiguous 1D tensors from the harness;
    # x0/x1 are fresh from .to(float32) so already contiguous.
    NUMEL = BATCH * CHANNELS * HW
    x0_flat = x0.view(NUMEL)
    x1_flat = x1.view(NUMEL)
    w0_f = arg1_1
    b0_f = arg2_1
    w1_f = arg4_1
    b1_f = arg5_1
    relu_flat = torch.empty(NUMEL, device=x0.device, dtype=torch.float32)
    bf16_flat = torch.empty(NUMEL, device=x0.device, dtype=torch.bfloat16)

    # BLOCK must divide NUMEL: NUMEL = 128 * C * HW.
    BLOCK = 128 if NUMEL % 128 == 0 else 64
    ct.launch(
        stream, (ct.cdiv(NUMEL, BLOCK), 1, 1),
        _dual_gn_epilogue_kernel,
        (x0_flat, x1_flat, mean0, invstd0, mean1, invstd1,
         w0_f, b0_f, w1_f, b1_f, relu_flat, bf16_flat,
         CHANNELS, reduction_total, GROUP_ELEMS, BLOCK),
    )

    relu = relu_flat.view(BATCH, CHANNELS, HEIGHT, WIDTH)
    bf16_out = bf16_flat.view(BATCH, CHANNELS, HEIGHT, WIDTH)

    # Reshape mean/invstd to (128, 32, 1, 1)
    mean0_4d = mean0.view(BATCH, GROUPS, 1, 1)
    invstd0_4d = invstd0.view(BATCH, GROUPS, 1, 1)
    mean1_4d = mean1.view(BATCH, GROUPS, 1, 1)
    invstd1_4d = invstd1.view(BATCH, GROUPS, 1, 1)
    # But reference reshapes: (128, 512, 1, 1). Wait — original outputs are
    # `getitem_1: f32[128, 512, 1, 1]`? No, from view (128, 32, 16, 1):
    # var_mean over [2, 3] with keepdim gives (128, 32, 1, 1). So expected shape is (128, 32, 1, 1).
    return mean0_4d, invstd0_4d, mean1_4d, invstd1_4d, relu, bf16_out


@oracle_impl(hardware="B200", point="ade33c47", CHANNELS=128, HEIGHT=4, WIDTH=4, GROUP_ELEMS=4)
@oracle_impl(hardware="B200", point="1fa56db2", CHANNELS=256, HEIGHT=2, WIDTH=2, GROUP_ELEMS=8)
@oracle_impl(hardware="B200", point="d227844f", CHANNELS=512, HEIGHT=1, WIDTH=1, GROUP_ELEMS=16)
def oracle_forward(inputs, *, CHANNELS, HEIGHT, WIDTH, GROUP_ELEMS):
    return _run(inputs, CHANNELS=CHANNELS, HEIGHT=HEIGHT, WIDTH=WIDTH, GROUP_ELEMS=GROUP_ELEMS)
