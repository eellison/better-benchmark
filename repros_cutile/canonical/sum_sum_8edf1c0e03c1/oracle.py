"""cuTile port of sum_sum_8edf1c0e03c1: DenseNet full-scope BN-backward + 8-way residual.

Triton fused: masked producer + two channel reductions + BN-backward epilogue
+ 8-way bf16 residual sum + a final bf16 add of the epilogue tail into it.

cuTile split: kernel1 produces the bf16 where(); torch performs the two
per-channel reductions and the sequential bf16 residual add; kernel2 runs the
BN-backward elementwise epilogue producing the full grad_out; torch does the
final bf16 add against the epilogue tail.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


# 1 / (N * H * W) = 1 / (4 * 28 * 28) = 1/3136.
INV_COUNT = 0.00031887755102040814


@ct.kernel
def _where_producer_kernel(
    mask_ptr,    # bf16[NUMEL]
    fill_ptr,    # bf16[1]
    grad_ptr,    # bf16[NUMEL]
    out_ptr,     # bf16[NUMEL]
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    mask_val = ct.load(mask_ptr, index=(pid,), shape=(BLOCK,))
    fill = ct.load(fill_ptr, index=(0,), shape=(1,))
    grad_val = ct.load(grad_ptr, index=(pid,), shape=(BLOCK,))
    zero_bf16 = ct.astype(ct.zeros((BLOCK,), dtype=ct.bfloat16), ct.bfloat16)
    le = mask_val <= zero_bf16
    out = ct.where(le, fill, grad_val)
    ct.store(out_ptr, index=(pid,), tile=out)


@ct.kernel
def _bn_backward_epilogue_kernel(
    where_ptr,       # bf16[NUMEL]
    centered_ptr,    # f32[NUMEL]
    mean_term_ptr,   # f32[NUMEL] (broadcast of sum_1 * inv_count)
    dot_coeff_ptr,   # f32[NUMEL] (broadcast of sum_2 * inv_count * invstd^2)
    out_scale_ptr,   # f32[NUMEL] (broadcast of invstd * weight)
    grad_out_ptr,    # bf16[NUMEL]
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    where_val = ct.astype(
        ct.load(where_ptr, index=(pid,), shape=(BLOCK,)), ct.float32
    )
    centered = ct.load(centered_ptr, index=(pid,), shape=(BLOCK,))
    mean_term = ct.load(mean_term_ptr, index=(pid,), shape=(BLOCK,))
    dot_coeff = ct.load(dot_coeff_ptr, index=(pid,), shape=(BLOCK,))
    out_scale = ct.load(out_scale_ptr, index=(pid,), shape=(BLOCK,))

    variance_term = centered * dot_coeff
    sub1 = where_val - variance_term
    sub2 = sub1 - mean_term
    grad_f32 = sub2 * out_scale
    ct.store(grad_out_ptr, index=(pid,), tile=ct.astype(grad_f32, ct.bfloat16))


@oracle_impl(hardware="B200", point="ccbdb456", BLOCK=1024)
def oracle_forward(inputs, *, BLOCK):
    (arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7,
     mask_src, fill, grad, activation, mean, invstd, weight) = inputs

    device = grad.device
    n = int(grad.shape[0])
    c = int(grad.shape[1])
    h = int(grad.shape[2])
    w = int(grad.shape[3])
    hw = h * w
    numel = n * c * hw
    stream = torch.cuda.current_stream()

    # ---- Kernel 1: where(le, fill, grad) producer over bf16 ----
    grad_c = grad.contiguous()
    mask_c = mask_src.contiguous()
    where_flat = torch.empty(numel, device=device, dtype=torch.bfloat16)
    fill_flat = fill.view(1)
    ct.launch(stream, (ct.cdiv(numel, BLOCK), 1, 1), _where_producer_kernel,
              (mask_c.view(numel), fill_flat, grad_c.view(numel),
               where_flat, BLOCK))
    where_val = where_flat.view(n, c, h, w)

    # ---- Torch: sequential bf16 8-way residual add on the [:, 224:256] slice ----
    slice_1 = arg0[:, 224:256, :, :]
    slice_2 = arg1[:, 224:256, :, :]
    slice_3 = arg2[:, 224:256, :, :]
    slice_4 = arg3[:, 224:256, :, :]
    slice_5 = arg4[:, 224:256, :, :]
    slice_6 = arg5[:, 224:256, :, :]
    slice_7 = arg6[:, 224:256, :, :]
    slice_8 = arg7[:, 224:256, :, :]
    residual = slice_1 + slice_2
    residual = residual + slice_3
    residual = residual + slice_4
    residual = residual + slice_5
    residual = residual + slice_6
    residual = residual + slice_7
    residual = residual + slice_8

    # ---- Torch: two per-channel reductions ----
    where_f32 = where_val.to(torch.float32)
    sum_1 = where_f32.sum(dim=[0, 2, 3])                     # f32[C]
    centered_f32 = activation.to(torch.float32) - mean       # (1,C,1,1) broadcast
    sum_2 = (where_f32 * centered_f32).sum(dim=[0, 2, 3])    # f32[C]

    mul_8 = sum_2 * invstd                                   # f32[C]

    mean_term_1d = sum_1 * INV_COUNT
    dot_coeff_1d = (sum_2 * INV_COUNT) * (invstd * invstd)
    out_scale_1d = invstd * weight

    def _bc(v):
        return v.view(1, c, 1, 1).expand(n, c, h, w).contiguous()

    mean_term_bc = _bc(mean_term_1d)
    dot_coeff_bc = _bc(dot_coeff_1d)
    out_scale_bc = _bc(out_scale_1d)
    centered_bc = centered_f32.expand(n, c, h, w).contiguous()

    # ---- Kernel 2: BN-backward elementwise epilogue -> bf16 grad_out ----
    grad_out_flat = torch.empty(numel, device=device, dtype=torch.bfloat16)
    ct.launch(stream, (ct.cdiv(numel, BLOCK), 1, 1), _bn_backward_epilogue_kernel,
              (where_flat, centered_bc.view(numel), mean_term_bc.view(numel),
               dot_coeff_bc.view(numel), out_scale_bc.view(numel),
               grad_out_flat, BLOCK))
    grad_out = grad_out_flat.view(n, c, h, w)

    # ---- Torch: final bf16 residual add of the [:, 224:256] tail ----
    slice_9 = grad_out[:, 224:256, :, :]
    add_7 = residual + slice_9

    return sum_1, mul_8, grad_out, add_7
