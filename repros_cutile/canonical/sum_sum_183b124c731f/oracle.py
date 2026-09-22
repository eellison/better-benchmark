"""cuTile port of sum_sum_183b124c731f: GhostNet BN backward.

Uses cuTile for the elementwise BN-backward epilogue. Per-channel reductions
are done in torch (graph-capturable).
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _bn_backward_epilogue_kernel(
    grad_ptr,        # bf16 [NUMEL]
    centered_ptr,    # f32  [NUMEL]
    mean1_ptr,       # f32  [NUMEL]
    variance_ptr,    # f32  [NUMEL]
    affine_ptr,      # f32  [NUMEL]
    out_ptr,         # bf16 [NUMEL]
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    grad_bf = ct.load(grad_ptr, index=(pid,), shape=(BLOCK,))
    grad_f = ct.astype(grad_bf, ct.float32)
    centered = ct.load(centered_ptr, index=(pid,), shape=(BLOCK,))
    mean1 = ct.load(mean1_ptr, index=(pid,), shape=(BLOCK,))
    variance = ct.load(variance_ptr, index=(pid,), shape=(BLOCK,))
    affine = ct.load(affine_ptr, index=(pid,), shape=(BLOCK,))

    centered_correction = centered * variance
    sub1 = grad_f - centered_correction
    sub2 = sub1 - mean1
    normalized = sub2 * affine
    ct.store(out_ptr, index=(pid,), tile=ct.astype(normalized, ct.bfloat16))


SCALE = 6.228077168367346e-07  # matches the Repro's baked-in constant


def _run(inputs, *, BLOCK):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1 = inputs
    n, c, h, w = int(arg0_1.shape[0]), int(arg0_1.shape[1]), int(arg0_1.shape[2]), int(arg0_1.shape[3])
    numel = n * c * h * w
    device = arg0_1.device
    inv_count = SCALE

    # Per-channel reductions via torch (matches Triton's per-channel sums).
    arg0_f = arg0_1.to(torch.float32)
    sum1 = arg0_f.sum(dim=[0, 2, 3])  # f32[C]
    centered_source_f = arg1_1.to(torch.float32) - arg2_1  # (1,C,1,1) -> broadcast
    sum2 = (arg0_f * centered_source_f).sum(dim=[0, 2, 3])  # f32[C]

    # Epilogue helper tensors
    mul1 = sum1 * inv_count  # f32[C]
    scale = arg3_1  # f32[C]
    mul2 = sum2 * inv_count
    mul3 = scale * scale
    mul4 = mul2 * mul3  # f32[C]
    affine_c = scale * arg4_1  # f32[C]
    mul8 = sum2 * arg3_1

    # Broadcast to (n,c,h,w)
    def _bc(vec_c):
        return vec_c.view(1, c, 1, 1).expand(n, c, h, w).contiguous()

    mean1_bc = _bc(mul1)
    variance_bc = _bc(mul4)
    affine_bc = _bc(affine_c)
    centered_bc = centered_source_f.expand(n, c, h, w).contiguous()

    out = torch.empty_strided(
        (n, c, h, w), (c * h * w, h * w, w, 1),  # match Triton (contiguous)
        device=device, dtype=torch.bfloat16,
    )
    # Actually strides in triton match input arg0_1.stride(). Let's mimic:
    out = torch.empty_strided(
        tuple(arg0_1.shape),
        tuple(arg0_1.stride()),
        device=device, dtype=torch.bfloat16,
    )

    # arg0_1 has channels-last stride: contiguous the input for the kernel
    arg0_flat = arg0_1.contiguous().view(numel)
    out_contig = torch.empty(numel, device=device, dtype=torch.bfloat16)

    stream = torch.cuda.current_stream()
    ct.launch(stream, (ct.cdiv(numel, BLOCK), 1, 1), _bn_backward_epilogue_kernel,
              (arg0_flat, centered_bc.view(numel), mean1_bc.view(numel),
               variance_bc.view(numel), affine_bc.view(numel),
               out_contig, BLOCK))

    # Move to strided output
    out_4d = out_contig.view(n, c, h, w)
    # Copy to output with correct strides (channels-last)
    out.copy_(out_4d)
    return sum1, mul8, out


@oracle_impl(hardware="B200", point="806629ba", BLOCK=1024)
@oracle_impl(hardware="B200", point="cbe8cdcd", BLOCK=1024)
@oracle_impl(hardware="B200", point="14bc6292", BLOCK=1024)
@oracle_impl(hardware="B200", point="bc4d6b09", BLOCK=1024)
@oracle_impl(hardware="B200", point="27df89f8", BLOCK=1024)
@oracle_impl(hardware="B200", point="b330737b", BLOCK=1024)
@oracle_impl(hardware="B200", point="99caacbc", BLOCK=1024)
@oracle_impl(hardware="B200", point="06e3c268", BLOCK=1024)
@oracle_impl(hardware="B200", point="7f09d5e1", BLOCK=1024)
@oracle_impl(hardware="B200", point="1c020ea5", BLOCK=1024)
@oracle_impl(hardware="B200", point="51768188", BLOCK=1024)
@oracle_impl(hardware="B200", point="0b7b8941", BLOCK=1024)
@oracle_impl(hardware="B200", point="4a86e94b", BLOCK=1024)
@oracle_impl(hardware="B200", point="797c887f", BLOCK=1024)
@oracle_impl(hardware="B200", point="fa717ebf", BLOCK=1024)
@oracle_impl(hardware="B200", point="64702d33", BLOCK=1024)
@oracle_impl(hardware="B200", point="fbff83c8", BLOCK=1024)
@oracle_impl(hardware="B200", point="c63234aa", BLOCK=1024)
@oracle_impl(hardware="B200", point="0ce2ed98", BLOCK=1024)
@oracle_impl(hardware="B200", point="a999a033", BLOCK=1024)
@oracle_impl(hardware="B200", point="59332feb", BLOCK=1024)
@oracle_impl(hardware="B200", point="0b3636fa", BLOCK=1024)
@oracle_impl(hardware="B200", point="05aad2cb", BLOCK=1024)
@oracle_impl(hardware="B200", point="92d95c9a", BLOCK=1024)
@oracle_impl(hardware="B200", point="feb664f7", BLOCK=1024)
@oracle_impl(hardware="B200", point="ba1f5763", BLOCK=1024)
@oracle_impl(hardware="B200", point="3ab290cb", BLOCK=1024)
@oracle_impl(hardware="B200", point="c37e4be5", BLOCK=1024)
@oracle_impl(hardware="B200", point="7ef2067a", BLOCK=1024)
@oracle_impl(hardware="B200", point="b38c4fd4", BLOCK=1024)
@oracle_impl(hardware="B200", point="b2f8b132", BLOCK=1024)
@oracle_impl(hardware="B200", point="bcf0f34b", BLOCK=1024)
@oracle_impl(hardware="B200", point="45be9d7c", BLOCK=1024)
@oracle_impl(hardware="B200", point="8f8e9775", BLOCK=1024)
@oracle_impl(hardware="B200", point="16d83ade", BLOCK=1024)
@oracle_impl(hardware="B200", point="3c4c0e95", BLOCK=1024)
@oracle_impl(hardware="B200", point="56236213", BLOCK=1024)
@oracle_impl(hardware="B200", point="73cdbfa1", BLOCK=1024)
@oracle_impl(hardware="B200", point="e001e4c0", BLOCK=1024)
@oracle_impl(hardware="B200", point="4248d2e3", BLOCK=1024)
@oracle_impl(hardware="B200", point="805fe5b3", BLOCK=1024)
@oracle_impl(hardware="B200", point="86433bd5", BLOCK=1024)
@oracle_impl(hardware="B200", point="819a1512", BLOCK=1024)
@oracle_impl(hardware="B200", point="8a1d8530", BLOCK=1024)
@oracle_impl(hardware="B200", point="40a8e8e1", BLOCK=1024)
def oracle_forward(inputs, *, BLOCK):
    return _run(inputs, BLOCK=BLOCK)
