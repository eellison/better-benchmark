"""cuTile port of sum_sum_a58657212bf4: MobileNetV2/EfficientNet BN backward.

Same as sum_sum_183b124c731f but with an initial `arg0 + arg1` bf16 add for
the grad tensor. Returns (sum_1, mul_8, bf16_out).
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


SCALE = 2.4912308673469386e-06  # matches the Repro's baked-in constant


@ct.kernel
def _bn_backward_epilogue_kernel(
    grad_ptr,        # f32  [NUMEL] - already added
    centered_ptr,    # f32  [NUMEL]
    mean1_ptr,       # f32  [NUMEL]
    variance_ptr,    # f32  [NUMEL]
    affine_ptr,      # f32  [NUMEL]
    out_ptr,         # bf16 [NUMEL]
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    grad_f = ct.load(grad_ptr, index=(pid,), shape=(BLOCK,))
    centered = ct.load(centered_ptr, index=(pid,), shape=(BLOCK,))
    mean1 = ct.load(mean1_ptr, index=(pid,), shape=(BLOCK,))
    variance = ct.load(variance_ptr, index=(pid,), shape=(BLOCK,))
    affine = ct.load(affine_ptr, index=(pid,), shape=(BLOCK,))

    centered_correction = centered * variance
    sub1 = grad_f - centered_correction
    sub2 = sub1 - mean1
    normalized = sub2 * affine
    ct.store(out_ptr, index=(pid,), tile=ct.astype(normalized, ct.bfloat16))


def _run(inputs, *, BLOCK):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1 = inputs
    n, c, h, w = arg0_1.shape
    n, c, h, w = int(n), int(c), int(h), int(w)
    numel = n * c * h * w
    device = arg0_1.device

    # grad = (arg0 + arg1) as bf16 -> f32
    grad_bf = arg0_1 + arg1_1  # bf16
    grad_f = grad_bf.to(torch.float32)

    sum1 = grad_f.sum(dim=[0, 2, 3])
    centered_source_f = arg2_1.to(torch.float32) - arg3_1
    sum2 = (grad_f * centered_source_f).sum(dim=[0, 2, 3])

    mul1 = sum1 * SCALE
    scale = arg4_1
    mul2 = sum2 * SCALE
    mul3 = scale * scale
    mul4 = mul2 * mul3
    affine_c = scale * arg5_1
    mul8 = sum2 * arg4_1

    def _bc(vec_c):
        return vec_c.view(1, c, 1, 1).expand(n, c, h, w).contiguous()

    mean1_bc = _bc(mul1)
    variance_bc = _bc(mul4)
    affine_bc = _bc(affine_c)
    centered_bc = centered_source_f.expand(n, c, h, w).contiguous()

    grad_flat = grad_f.contiguous().view(numel)
    out_contig = torch.empty(numel, device=device, dtype=torch.bfloat16)

    stream = torch.cuda.current_stream()
    ct.launch(stream, (ct.cdiv(numel, BLOCK), 1, 1), _bn_backward_epilogue_kernel,
              (grad_flat, centered_bc.view(numel), mean1_bc.view(numel),
               variance_bc.view(numel), affine_bc.view(numel),
               out_contig, BLOCK))

    out_4d = out_contig.view(n, c, h, w)
    return sum1, mul8, out_4d


_POINTS = [
    "f3fd9151", "65b876e3", "1b9feebb", "3726f4ca", "9793b43e", "3d50e493",
    "399aa3e2", "39a9326e", "86ef280c", "b6f518ab", "a45e6340", "da738408",
    "6f1023fc", "b5264010", "315c2b3e", "0dc5b6bd", "864b3c6f", "cf15f756",
    "409c8bd3", "727b7028", "4d254913", "3edd6c00", "ee318906", "1592ce3d",
    "ebb56431", "f63ebc76", "f35ade00",
]


def _apply(fn):
    for p in _POINTS:
        fn = oracle_impl(hardware="B200", point=p, BLOCK=1024)(fn)
    return fn


@_apply
def oracle_forward(inputs, *, BLOCK):
    return _run(inputs, BLOCK=BLOCK)
