"""cuTile port of sum_sum_cd8694c00507: SiLU + BN-backward.

Strategy: cuTile kernel 1 emits the SiLU-gradient producer as bf16, torch
computes the two per-channel reductions, cuTile kernel 2 runs the BN-backward
epilogue producing bf16 output.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


REDUCE_SCALE = 4.76837158203125e-07


@ct.kernel
def _silu_producer_kernel(
    grad_ptr,      # bf16[NUMEL]
    x_ptr,         # bf16[NUMEL]
    mean_ptr,      # f32[NUMEL] broadcast (1,C,1,1) -> (N,C,H,W)
    invstd_ptr,    # f32[NUMEL] broadcast
    weight_ptr,    # f32[NUMEL] broadcast
    bias_ptr,      # f32[NUMEL] broadcast
    out_ptr,       # bf16[NUMEL]
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    grad = ct.astype(ct.load(grad_ptr, index=(pid,), shape=(BLOCK,)), ct.float32)
    x = ct.astype(ct.load(x_ptr, index=(pid,), shape=(BLOCK,)), ct.float32)
    mean = ct.load(mean_ptr, index=(pid,), shape=(BLOCK,))
    invstd = ct.load(invstd_ptr, index=(pid,), shape=(BLOCK,))
    weight = ct.load(weight_ptr, index=(pid,), shape=(BLOCK,))
    bias = ct.load(bias_ptr, index=(pid,), shape=(BLOCK,))

    centered = x - mean
    affine = centered * invstd * weight + bias
    # Round through bf16 to match eager's `convert_element_type_1 = ...to(bf16)`
    # followed by `.to(f32)` for sigmoid.
    affine_bf = ct.astype(ct.astype(affine, ct.bfloat16), ct.float32)

    # sigmoid = 1 / (1 + exp(-x))
    sigmoid = 1.0 / (1.0 + ct.exp(-affine_bf))
    grad_sigmoid = grad * sigmoid
    tail = affine_bf * (1.0 - sigmoid) + 1.0
    producer_f = grad_sigmoid * tail
    ct.store(out_ptr, index=(pid,), tile=ct.astype(producer_f, ct.bfloat16))


@ct.kernel
def _epilogue_kernel(
    producer_ptr,   # bf16[NUMEL]
    centered_ptr,   # f32[NUMEL]
    mean_term_ptr,  # f32[NUMEL]
    dot_coeff_ptr,  # f32[NUMEL]
    out_scale_ptr,  # f32[NUMEL]
    out_ptr,        # bf16[NUMEL]
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    producer = ct.astype(ct.load(producer_ptr, index=(pid,), shape=(BLOCK,)), ct.float32)
    centered = ct.load(centered_ptr, index=(pid,), shape=(BLOCK,))
    mean_term = ct.load(mean_term_ptr, index=(pid,), shape=(BLOCK,))
    dot_coeff = ct.load(dot_coeff_ptr, index=(pid,), shape=(BLOCK,))
    out_scale = ct.load(out_scale_ptr, index=(pid,), shape=(BLOCK,))
    out_f = (producer - centered * dot_coeff - mean_term) * out_scale
    ct.store(out_ptr, index=(pid,), tile=ct.astype(out_f, ct.bfloat16))


def _run(inputs, *, BLOCK):
    grad, x, mean_1c11, invstd_1c11, weight, bias = inputs
    device = grad.device
    n, c, h, w = int(x.shape[0]), int(x.shape[1]), int(x.shape[2]), int(x.shape[3])
    numel = n * c * h * w

    def _bc(vec_c):
        return vec_c.view(1, c, 1, 1).expand(n, c, h, w).contiguous()

    # Materialize NCHW-contiguous copies for the kernels to walk in flat order.
    x_nchw = x.contiguous()
    grad_nchw = grad.contiguous()
    mean_bc = _bc(mean_1c11.view(c))
    invstd_bc = _bc(invstd_1c11.view(c))
    weight_bc = _bc(weight)
    bias_bc = _bc(bias)

    producer_flat = torch.empty(numel, device=device, dtype=torch.bfloat16)
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(numel, BLOCK), 1, 1),
        _silu_producer_kernel,
        (
            grad_nchw.view(numel),
            x_nchw.view(numel),
            mean_bc.view(numel),
            invstd_bc.view(numel),
            weight_bc.view(numel),
            bias_bc.view(numel),
            producer_flat,
            BLOCK,
        ),
    )
    producer_nchw = producer_flat.view(n, c, h, w)

    # Torch handles the two per-channel reductions (fp32 accumulator).
    producer_f = producer_nchw.to(torch.float32)
    centered_f = x_nchw.to(torch.float32) - mean_1c11  # broadcasts (1,C,1,1)
    sum_1 = producer_f.sum(dim=[0, 2, 3])
    sum_2 = (producer_f * centered_f).sum(dim=[0, 2, 3])
    invstd_1d = invstd_1c11.view(c)
    mul_13 = sum_2 * invstd_1d

    mean_term_1d = sum_1 * REDUCE_SCALE
    dot_coeff_1d = (sum_2 * REDUCE_SCALE) * (invstd_1d * invstd_1d)
    out_scale_1d = invstd_1d * weight

    mean_term_bc = _bc(mean_term_1d)
    dot_coeff_bc = _bc(dot_coeff_1d)
    out_scale_bc = _bc(out_scale_1d)
    centered_bc = centered_f.expand(n, c, h, w).contiguous()

    # Output channels-last (matches the Triton oracle's returned strides).
    out_channels_last = torch.empty_strided(
        (n, c, h, w),
        (c * h * w, 1, w * c, c),
        device=device,
        dtype=torch.bfloat16,
    )
    out_flat = torch.empty(numel, device=device, dtype=torch.bfloat16)
    ct.launch(
        stream,
        (ct.cdiv(numel, BLOCK), 1, 1),
        _epilogue_kernel,
        (
            producer_flat,
            centered_bc.view(numel),
            mean_term_bc.view(numel),
            dot_coeff_bc.view(numel),
            out_scale_bc.view(numel),
            out_flat,
            BLOCK,
        ),
    )
    out_channels_last.copy_(out_flat.view(n, c, h, w))

    return sum_1, mul_13, out_channels_last


_POINTS = [
    "57aa57bf", "4a93118a", "f444ad5f", "0312171c", "65ca7fc0", "315c8b2f",
    "f11d7f2c", "04100efe", "9fc1d6b0", "a6ded16e", "0466d5cd", "6ef366bb",
    "def98aed", "d6507bc1", "d6acaf93", "787e6d78", "1bde3bc1", "14e1266b",
    "168f8cc2",
]


def _register(fn):
    for point in _POINTS:
        fn = oracle_impl(hardware="B200", point=point, BLOCK=1024)(fn)
    return fn


@_register
def oracle_forward(inputs, *, BLOCK):
    return _run(inputs, BLOCK=BLOCK)
