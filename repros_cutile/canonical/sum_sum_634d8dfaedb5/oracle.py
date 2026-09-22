"""cuTile port of sum_sum_634d8dfaedb5: SqueezeNet dual slice/where/sum.

Reference (bf16 arg0=[32,512,13,13], b8 arg1=[32,256,13,13], bf16 arg2=[], b8 arg3=[32,256,13,13]):
  slice_1 = arg0[:, 0:256]
  slice_2 = arg0[:, 256:512]
  where_r = where(arg1, arg2, slice_2)
  where_l = where(arg3, arg2, slice_1)
  sum_r = sum(where_r, dim=[0,2,3]).to(f32)
  sum_l = sum(where_l, dim=[0,2,3]).to(f32)
  return (where_r, sum_r, where_l, sum_l)

One kernel per channel c: load 32*169 elements of slice_1/slice_2 and both masks
(rounded up to power-of-2 K=8192), compute the two `where` values, store them,
sum-reduce.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


N = 32
C = 256
HW = 169
K = N * HW  # 32 * 169 = 5408
BLOCK_K = 8192  # next power of two > K


@ct.kernel
def _slice_where_sum_kernel(
    slice_ptr,      # bf16 [C, K] (row-major on a per-channel narrow view)
    mask_ptr,       # b8 [C, K]
    fill_ptr,       # bf16 scalar (viewed as len-1 tile)
    where_out_ptr,  # bf16 [C, K]
    sum_out_ptr,    # f32 [C]
    K_: ct.Constant[int],
    BLOCK_K_: ct.Constant[int],
):
    c = ct.bid(0)
    slice_tile = ct.load(
        slice_ptr, index=(c, 0), shape=(1, BLOCK_K_),
        padding_mode=ct.PaddingMode.ZERO,
    )
    mask_tile = ct.load(
        mask_ptr, index=(c, 0), shape=(1, BLOCK_K_),
        padding_mode=ct.PaddingMode.ZERO,
    )
    fill_scalar = ct.load(fill_ptr, index=(0,), shape=(1,))
    fill_2d = ct.reshape(fill_scalar, (1, 1))

    keep = mask_tile != ct.zeros(shape=(1, BLOCK_K_), dtype=ct.bool_)
    where_val = ct.where(keep, fill_2d, slice_tile)
    ct.store(where_out_ptr, index=(c, 0), tile=where_val)

    where_f32 = ct.astype(where_val, ct.float32)
    col_idx = ct.arange(BLOCK_K_, dtype=ct.int32)
    active = ct.reshape(col_idx < K_, (1, BLOCK_K_))
    reduced = ct.where(active, where_f32, 0.0)
    total = ct.sum(reduced)
    ct.store(sum_out_ptr, index=(c,), tile=ct.reshape(total, (1,)))


@oracle_impl(hardware="B200", point="220691ab")
def oracle_forward(inputs, **_kwargs):
    arg0_1, arg1_1, arg2_1, arg3_1 = inputs
    # arg0_1: bf16 [32, 512, 13, 13]
    # arg1_1, arg3_1: b8 [32, 256, 13, 13]
    # arg2_1: bf16 [] (scalar)

    device = arg0_1.device

    # View slice_1 = arg0[:, 0:256] and slice_2 = arg0[:, 256:512] as [C, N*HW]
    # via permutation.  Original NCHW layout: element (n, c, h, w) at offset
    # n*C_full*HW + c*HW + h*W + w. We want [C, N*HW]: same C rows, spatial +
    # batch flattened. That requires N * HW contiguity per channel, which is
    # NOT the native layout (channels are contiguous within a batch).
    # Simplest: transpose to [C, N, H, W] via permute and copy contiguous.
    slice_r = arg0_1[:, C:2 * C, :, :].permute(1, 0, 2, 3).contiguous().view(C, N * HW)
    slice_l = arg0_1[:, 0:C, :, :].permute(1, 0, 2, 3).contiguous().view(C, N * HW)
    mask_r = arg1_1.permute(1, 0, 2, 3).contiguous().view(C, N * HW)
    mask_l = arg3_1.permute(1, 0, 2, 3).contiguous().view(C, N * HW)

    where_r_flat = torch.empty((C, N * HW), device=device, dtype=torch.bfloat16)
    where_l_flat = torch.empty((C, N * HW), device=device, dtype=torch.bfloat16)
    sum_r = torch.empty((C,), device=device, dtype=torch.float32)
    sum_l = torch.empty((C,), device=device, dtype=torch.float32)

    # Because BLOCK_K > K, allocate padded scratch for the stored `where` and
    # narrow after (K = 5408, BLOCK_K = 8192).
    where_r_pad = torch.empty((C, BLOCK_K), device=device, dtype=torch.bfloat16)
    where_l_pad = torch.empty((C, BLOCK_K), device=device, dtype=torch.bfloat16)

    fill_view = arg2_1.view(1)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (C, 1, 1), _slice_where_sum_kernel,
        (slice_r, mask_r, fill_view, where_r_pad, sum_r, K, BLOCK_K),
    )
    ct.launch(
        stream, (C, 1, 1), _slice_where_sum_kernel,
        (slice_l, mask_l, fill_view, where_l_pad, sum_l, K, BLOCK_K),
    )

    where_r_flat.copy_(where_r_pad.narrow(1, 0, N * HW))
    where_l_flat.copy_(where_l_pad.narrow(1, 0, N * HW))

    # Reshape [C, N*HW] back to [N, C, H, W] via inverse permute.
    where_r = where_r_flat.view(C, N, 13, 13).permute(1, 0, 2, 3).contiguous()
    where_l = where_l_flat.view(C, N, 13, 13).permute(1, 0, 2, 3).contiguous()

    return where_r, sum_r, where_l, sum_l
