"""cuTile port of sum_sum_03624e225234: SqueezeNet channels-last where + sum.

Inputs:
  arg0: bf16 [32, 128, 55, 55] channels-last
  arg1: b8  [32, 64, 55, 55] channels-last
  arg2: bf16 scalar (fill)
  arg3: b8  [32, 64, 55, 55] channels-last

Structure:
  slice_1 = arg0[:, 0:64]     (channels-last view)
  slice_2 = arg0[:, 64:128]   (channels-last view)
  where   = where(arg1, arg2, slice_2)   -> bf16 [32, 64, 55, 55]
  where_1 = where(arg3, arg2, slice_1)   -> bf16 [32, 64, 55, 55]
  sum_1   = bf16 sum(where, [0, 2, 3])   -> f32 [64]  (sum done in bf16!)
  sum_2   = bf16 sum(where_1, [0, 2, 3]) -> f32 [64]

Approach:
  cuTile pointwise kernel writes both where outputs (channels-last) via flat
  NHWC iteration. Torch computes the two per-channel bf16 sums.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _dual_where_kernel(
    input_ptr,       # bf16 [N, H, W, 128] flat NHWC (physical)
    right_mask_ptr,  # b8   [N, H, W, 64] flat NHWC (arg1)
    left_mask_ptr,   # b8   [N, H, W, 64] flat NHWC (arg3)
    fill_ptr,        # bf16 [1]
    right_out_ptr,   # bf16 [N, H, W, 64] flat NHWC (where)
    left_out_ptr,    # bf16 [N, H, W, 64] flat NHWC (where_1)
    C_HALF: ct.Constant[int],
    INPUT_C: ct.Constant[int],
    TOTAL_HALF: ct.Constant[int],
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    # idxs in [0, TOTAL_HALF), maps into the (N, HW, C_HALF) output layout.
    idxs = ct.arange(BLOCK, dtype=ct.int32) + pid * BLOCK
    # For NHWC contiguous, out flat idx k -> (nhw = k // C_HALF, c = k % C_HALF).
    nhw = idxs // C_HALF
    c_off = idxs - nhw * C_HALF
    # Input flat idx (128-wide channels): nhw*INPUT_C + slice_offset + c_off.
    # slice_1 (input[:, 0:64]) is c_off + 0
    # slice_2 (input[:, 64:128]) is c_off + 64
    slice1_off = nhw * INPUT_C + c_off
    slice2_off = nhw * INPUT_C + c_off + C_HALF

    zero_bf = ct.astype(0.0, ct.bfloat16)
    zero_b8 = ct.astype(0, ct.bool_)
    slice1 = ct.gather(input_ptr, slice1_off, padding_value=zero_bf)
    slice2 = ct.gather(input_ptr, slice2_off, padding_value=zero_bf)
    right_mask = ct.load(right_mask_ptr, index=(pid,), shape=(BLOCK,))
    left_mask = ct.load(left_mask_ptr, index=(pid,), shape=(BLOCK,))
    fill_v = ct.load(fill_ptr, index=(0,), shape=(1,))
    fill_bc = ct.broadcast_to(fill_v, (BLOCK,))

    # where: right_mask ? fill : slice2
    right_where = ct.where(right_mask, fill_bc, slice2)
    # where_1: left_mask ? fill : slice1
    left_where = ct.where(left_mask, fill_bc, slice1)

    ct.store(right_out_ptr, index=(pid,), tile=right_where)
    ct.store(left_out_ptr, index=(pid,), tile=left_where)


N = 32
C = 64
INPUT_C = 128
H = 55
W = 55
HW = H * W
TOTAL_HALF = N * HW * C  # 32*3025*64 = 6,195,200


@oracle_impl(hardware="B200", point="185fdb55")
def oracle_forward(inputs):
    input_tensor, right_mask, fill, left_mask = inputs
    device = input_tensor.device

    # input_tensor stride (387200, 1, 7040, 128) -> memory NHWC
    # right_mask stride (193600, 1, 3520, 64) -> memory NHWC (b8 elements)
    input_flat = torch.as_strided(input_tensor, (N * HW * INPUT_C,), (1,))
    right_mask_flat = torch.as_strided(right_mask, (N * HW * C,), (1,))
    left_mask_flat = torch.as_strided(left_mask, (N * HW * C,), (1,))
    fill_view = fill.view(1)

    right_out_flat = torch.empty(TOTAL_HALF, device=device, dtype=torch.bfloat16)
    left_out_flat = torch.empty(TOTAL_HALF, device=device, dtype=torch.bfloat16)

    # BLOCK must divide TOTAL_HALF cleanly.
    # TOTAL_HALF = 32*55*55*64 = 6,195,200. Factor: 6195200 = 2^7*5^2*11^2*...
    # 6195200 / 256 = 24200 (exact).
    BLOCK = 256
    stream = torch.cuda.current_stream()
    ct.launch(
        stream, ((TOTAL_HALF + BLOCK - 1) // BLOCK, 1, 1), _dual_where_kernel,
        (input_flat, right_mask_flat, left_mask_flat, fill_view,
         right_out_flat, left_out_flat,
         C, INPUT_C, TOTAL_HALF, BLOCK),
    )

    # Reinterpret NHWC memory as (N, C, H, W) with channels-last strides.
    right_out = torch.as_strided(
        right_out_flat, (N, C, H, W), (C * HW, 1, W * C, C)
    )
    left_out = torch.as_strided(
        left_out_flat, (N, C, H, W), (C * HW, 1, W * C, C)
    )

    # Sums in bf16 then f32 cast (matches eager).
    right_sum_bf = right_out.sum(dim=[0, 2, 3], dtype=torch.bfloat16)
    left_sum_bf = left_out.sum(dim=[0, 2, 3], dtype=torch.bfloat16)
    right_sum = right_sum_bf.to(torch.float32)
    left_sum = left_sum_bf.to(torch.float32)

    return right_out, right_sum, left_out, left_sum
