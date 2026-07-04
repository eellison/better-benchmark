"""cuTile port of sum_sum_4e92555c35a0: DenseNet BN backward with slice sum + relu mask."""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


SCALE = 0.00031887755102040814
SLICE_START = 256
SLICE_END = 288


@ct.kernel
def _bn_backward_epilogue_kernel(
    where_ptr,
    centered_ptr,
    mean1_ptr,
    variance_ptr,
    affine_ptr,
    out_ptr,
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    where_bf = ct.load(where_ptr, index=(pid,), shape=(BLOCK,))
    where_f = ct.astype(where_bf, ct.float32)
    centered = ct.load(centered_ptr, index=(pid,), shape=(BLOCK,))
    mean1 = ct.load(mean1_ptr, index=(pid,), shape=(BLOCK,))
    variance = ct.load(variance_ptr, index=(pid,), shape=(BLOCK,))
    affine = ct.load(affine_ptr, index=(pid,), shape=(BLOCK,))

    centered_correction = centered * variance
    sub1 = where_f - centered_correction
    sub2 = sub1 - mean1
    normalized = sub2 * affine
    ct.store(out_ptr, index=(pid,), tile=ct.astype(normalized, ct.bfloat16))


def _run(inputs, *, BLOCK):
    (arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1,
     arg9_1, arg10_1, arg11_1, arg12_1, arg13_1) = inputs
    device = arg0_1.device

    # 7 slice adds (bf16)
    slice_add = arg0_1[:, SLICE_START:SLICE_END, :, :] + arg1_1[:, SLICE_START:SLICE_END, :, :]
    slice_add = slice_add + arg2_1[:, SLICE_START:SLICE_END, :, :]
    slice_add = slice_add + arg3_1[:, SLICE_START:SLICE_END, :, :]
    slice_add = slice_add + arg4_1[:, SLICE_START:SLICE_END, :, :]
    slice_add = slice_add + arg5_1[:, SLICE_START:SLICE_END, :, :]
    slice_add = slice_add + arg6_1[:, SLICE_START:SLICE_END, :, :]

    # where(le, arg8, arg9)
    le = arg7_1 <= 0
    where_out = torch.where(le, arg8_1, arg9_1)  # bf16
    where_f = where_out.to(torch.float32)
    n, c, h, w = where_out.shape
    n, c, h, w = int(n), int(c), int(h), int(w)
    numel = n * c * h * w

    sum1 = where_f.sum(dim=[0, 2, 3])
    centered_source_f = arg10_1.to(torch.float32) - arg11_1
    sum2 = (where_f * centered_source_f).sum(dim=[0, 2, 3])

    mul1 = sum1 * SCALE
    mul2 = sum2 * SCALE
    mul3 = arg12_1 * arg12_1
    mul4 = mul2 * mul3
    affine_c = arg12_1 * arg13_1
    mul8 = sum2 * arg12_1

    def _bc(vec_c):
        return vec_c.view(1, c, 1, 1).expand(n, c, h, w).contiguous()

    mean1_bc = _bc(mul1)
    variance_bc = _bc(mul4)
    affine_bc = _bc(affine_c)
    centered_bc = centered_source_f.expand(n, c, h, w).contiguous()

    where_flat = where_out.contiguous().view(numel)
    out_contig = torch.empty(numel, device=device, dtype=torch.bfloat16)

    stream = torch.cuda.current_stream()
    ct.launch(stream, (ct.cdiv(numel, BLOCK), 1, 1), _bn_backward_epilogue_kernel,
              (where_flat, centered_bc.view(numel), mean1_bc.view(numel),
               variance_bc.view(numel), affine_bc.view(numel),
               out_contig, BLOCK))

    out_4d = out_contig.view(n, c, h, w)
    slice_8 = out_4d[:, SLICE_START:SLICE_END, :, :]
    add_6 = slice_add + slice_8
    return sum1, mul8, out_4d, add_6


@oracle_impl(hardware="B200", point="9be388cf", BLOCK=1024)
def oracle_forward(inputs, *, BLOCK):
    return _run(inputs, BLOCK=BLOCK)
