"""cuTile port of pointwise_75d57ea4f44c: XLNet dual bf16 broadcast-add."""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _dual_broadcast_add_bf16_kernel(
    x_ptr,      # 1D [n_elements] bf16
    bias0_ptr,  # 1D [BIAS_LEN] f32 (broadcast tiles of length BLOCK)
    bias1_ptr,  # 1D [BIAS_LEN] f32
    out0_ptr,   # 1D [n_elements] bf16
    out1_ptr,   # 1D [n_elements] bf16
    BLOCK: ct.Constant[int],
    BIAS_LEN: ct.Constant[int],
):
    pid = ct.bid(0)
    x = ct.load(x_ptr, index=(pid,), shape=(BLOCK,))
    x_f = ct.astype(x, ct.float32)
    # BLOCK==BIAS_LEN so each tile mode's bias tile is just load the whole thing.
    bias0 = ct.load(bias0_ptr, index=(0,), shape=(BIAS_LEN,))
    bias1 = ct.load(bias1_ptr, index=(0,), shape=(BIAS_LEN,))
    ct.store(out0_ptr, index=(pid,), tile=ct.astype(x_f + bias0, ct.bfloat16))
    ct.store(out1_ptr, index=(pid,), tile=ct.astype(x_f + bias1, ct.bfloat16))


@oracle_impl(hardware="B200", point="798a086d", BLOCK=1024)
def oracle_forward(inputs, *, BLOCK):
    arg0_1, arg1_1, arg2_1, _s0, _s1, _s2, _s3 = inputs
    # arg0 is [8192, 1024] bf16
    # bias tiles are [16, 64] f32 = length 1024 each when flattened
    view_2 = torch.empty_strided(
        tuple(_s2),
        (64, 16384, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    view_3 = torch.empty_strided(
        tuple(_s3),
        (64, 16384, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    n_elements = arg0_1.numel()

    # view_2/view_3 are strided views into 8192x1024 storage; work directly on
    # backing storage via as_strided since the strides (64, 16384, 1) form
    # the transpose of 8192x1024 contiguous data.
    # But those view_2/view_3 tensors are strided — cuTile can handle strides,
    # but the "linear" mapping to storage is not contiguous. Simpler: allocate
    # a plain contiguous [8192, 1024] backing buffer, then have oracle_forward
    # return the strided view of that same storage.
    backing_2 = torch.empty_strided(
        (8192, 1024), (1024, 1), device=arg0_1.device, dtype=torch.bfloat16)
    backing_3 = torch.empty_strided(
        (8192, 1024), (1024, 1), device=arg0_1.device, dtype=torch.bfloat16)

    bias0_flat = arg1_1.reshape(-1)
    bias1_flat = arg2_1.reshape(-1)
    x_flat = arg0_1.reshape(-1)
    out0_flat = backing_2.reshape(-1)
    out1_flat = backing_3.reshape(-1)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (n_elements // BLOCK, 1, 1),
        _dual_broadcast_add_bf16_kernel,
        (x_flat, bias0_flat, bias1_flat, out0_flat, out1_flat, BLOCK, 1024),
    )

    # Re-view backing storage as the target 3D shape/stride, matching Repro's expected layout.
    # view_2 shape (256, 512, 64) with stride (64, 16384, 1) is the permute-of-[a,b,c,d]
    # where the flat storage is [8192, 1024] contiguous.
    view_2_final = torch.as_strided(backing_2, tuple(_s2), (64, 16384, 1))
    view_3_final = torch.as_strided(backing_3, tuple(_s3), (64, 16384, 1))
    return (
        view_2_final,
        view_3_final,
        view_3_final.permute(0, 2, 1),
        view_2_final.permute(0, 2, 1),
    )
