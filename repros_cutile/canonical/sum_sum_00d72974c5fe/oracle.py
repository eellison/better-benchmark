"""cuTile port of sum_sum_00d72974c5fe: DenseNet BN backward tail.

The reference computes `where(le, fill, source)` masked producer, two channel
reductions, then a BN-backward epilogue with a bf16 rounding boundary
before the residual add. cuTile fp32 arithmetic is already round-to-nearest,
so Triton's PTX helpers become plain operators.

Port strategy: use cuTile for the elementwise BN-backward epilogue producer
(where + subtract + multiply + bf16 rounding), let torch handle the two
per-channel reductions. Channels=112 non-pow2 so tile over flat NCHW.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _where_producer_kernel(
    mask_input_ptr,   # bf16[NUMEL]
    fill_ptr,         # bf16[1]
    source_ptr,       # bf16[NUMEL]
    out_ptr,          # bf16[NUMEL]
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    mask_input = ct.load(mask_input_ptr, index=(pid,), shape=(BLOCK,))
    fill = ct.load(fill_ptr, index=(0,), shape=(1,))
    source = ct.load(source_ptr, index=(pid,), shape=(BLOCK,))
    zero_bf16 = ct.astype(ct.zeros((BLOCK,), dtype=ct.bfloat16), ct.bfloat16)
    le = mask_input <= zero_bf16
    out = ct.where(le, fill, source)
    ct.store(out_ptr, index=(pid,), tile=out)


@ct.kernel
def _bn_backward_epilogue_kernel(
    where_ptr,       # bf16[NUMEL]
    centered_ptr,    # f32[NUMEL] (arg4 - arg5 broadcast) — per element
    mean1_ptr,       # f32[NUMEL] (per-element broadcast of sum1*inv_count)
    variance_ptr,    # f32[NUMEL] (per-element broadcast of sum2*inv_count*scale*scale)
    affine_ptr,      # f32[NUMEL] (per-element broadcast of scale * affine_weight)
    residual_ptr,    # bf16[NUMEL]
    add_out_ptr,     # bf16[NUMEL]
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    where_val = ct.astype(
        ct.load(where_ptr, index=(pid,), shape=(BLOCK,)), ct.float32
    )
    centered = ct.load(centered_ptr, index=(pid,), shape=(BLOCK,))
    mean1 = ct.load(mean1_ptr, index=(pid,), shape=(BLOCK,))
    variance = ct.load(variance_ptr, index=(pid,), shape=(BLOCK,))
    affine = ct.load(affine_ptr, index=(pid,), shape=(BLOCK,))
    residual = ct.astype(
        ct.load(residual_ptr, index=(pid,), shape=(BLOCK,)), ct.float32
    )

    centered_scaled = centered * variance
    sub1 = where_val - centered_scaled
    sub2 = sub1 - mean1
    normalized = sub2 * affine
    normalized_bf16_f32 = ct.astype(ct.astype(normalized, ct.bfloat16), ct.float32)
    add_val = residual + normalized_bf16_f32
    ct.store(add_out_ptr, index=(pid,), tile=ct.astype(add_val, ct.bfloat16))


def _run(inputs, *, BLOCK):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1 = inputs
    n, c, h, w = int(arg1_1.shape[0]), int(arg1_1.shape[1]), int(arg1_1.shape[2]), int(arg1_1.shape[3])
    residual_c = int(arg0_1.shape[1])
    numel = n * c * h * w
    hw = h * w
    # The captured graph baked in this scalar (1/(128*32*32)) regardless
    # of the input shape — matches how the reference module was traced.
    inv_count = 7.62939453125e-06

    device = arg1_1.device

    # ---- Kernel 1: where producer ----
    where_flat = torch.empty(numel, device=device, dtype=torch.bfloat16)
    arg1_flat = arg1_1.contiguous().view(numel)
    arg3_flat = arg3_1.contiguous().view(numel)
    fill_flat = arg2_1.view(1)
    stream = torch.cuda.current_stream()

    # BLOCK must divide numel. numel = 128 * 112 * H * W. For BLOCK=1024: 128*112*1024/1024=128*112=14336. Ok if HW*4 divisible? Let's check: HW=1024, HW=256, HW=64. n*c*HW = 128*112*HW.
    # 128*112*1024 = 14,680,064. /1024 = 14336. OK.
    # 128*112*256 = 3,670,016. /1024 = 3584. OK.
    # 128*112*64 = 917,504. /1024 = 896. OK.
    ct.launch(stream, (ct.cdiv(numel, BLOCK), 1, 1), _where_producer_kernel,
              (arg1_flat, fill_flat, arg3_flat, where_flat, BLOCK))

    where_val_4d = where_flat.view(n, c, h, w)

    # ---- Reductions via torch (matches the Triton kernel's per-channel sums) ----
    where_f32 = where_val_4d.to(torch.float32)
    sum1 = where_f32.sum(dim=[0, 2, 3])                        # f32[C]
    centered_source_f32 = arg4_1.to(torch.float32) - arg5_1    # (1,C,1,1) -> broadcasts to (N,C,H,W)
    sum2 = (where_f32 * centered_source_f32).sum(dim=[0, 2, 3])  # f32[C]

    mul8 = sum2 * arg6_1                                        # f32[C]

    # ---- Build broadcast tensors for epilogue ----
    mul1 = sum1 * inv_count                                     # f32[C]
    scale = arg6_1                                              # f32[C]
    mul2 = sum2 * inv_count                                     # f32[C]
    mul3 = scale * scale
    mul4 = mul2 * mul3                                          # f32[C]
    affine_c = scale * arg7_1                                   # f32[C]

    # Broadcast to (N,C,H,W)
    def _bc(vec_c):
        # (C,) -> (1, C, 1, 1) -> broadcast (N, C, H, W)
        return vec_c.view(1, c, 1, 1).expand(n, c, h, w).contiguous()

    mean1_bc = _bc(mul1)
    variance_bc = _bc(mul4)
    affine_bc = _bc(affine_c)
    centered_bc = centered_source_f32.expand(n, c, h, w).contiguous()

    # ---- Kernel 2: BN-backward epilogue producing bf16 add ----
    add_out_4d = torch.empty_strided(
        (n, c, h, w),
        (c * hw, hw, w, 1),
        device=device, dtype=torch.bfloat16,
    )
    # slice_1 = arg0[:, 16:16+112, :, :] is the residual for the add.
    slice1 = arg0_1[:, 16:16 + c, :, :].contiguous()
    slice1_flat = slice1.view(numel)
    add_out_flat = add_out_4d.view(numel)

    ct.launch(stream, (ct.cdiv(numel, BLOCK), 1, 1), _bn_backward_epilogue_kernel,
              (where_flat, centered_bc.view(numel), mean1_bc.view(numel),
               variance_bc.view(numel), affine_bc.view(numel),
               slice1_flat, add_out_flat, BLOCK))

    slice2 = add_out_4d[:, :16, :, :]
    return sum1, mul8, add_out_4d, slice2


@oracle_impl(hardware="B200", point="0b002951", BLOCK=1024)
@oracle_impl(hardware="B200", point="21952d3a", BLOCK=1024)
@oracle_impl(hardware="B200", point="75c39973", BLOCK=1024)
def oracle_forward(inputs, *, BLOCK):
    return _run(inputs, BLOCK=BLOCK)
