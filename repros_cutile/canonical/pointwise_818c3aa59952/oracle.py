"""cuTile port of pointwise_818c3aa59952 (ALGEBRAIC_ELIMINATION): BN inference
affine. Per-element compute `(x - mean) * invstd * weight + bias` over
channels-last or contiguous NCHW input, with `invstd = 1/sqrt(var + eps)`
hoisted to a per-channel producer.

Handles non-power-of-2 C by gathering per-channel params using
`ct.gather` inside a flat 1D pointwise kernel, and using `ct.scatter` with a
mask for the tail tile so OOB writes never occur.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


EPS = 1.0e-5


@ct.kernel
def _invstd_kernel(
    var_ptr,
    invstd_ptr,
    C: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    var = ct.astype(
        ct.load(var_ptr, index=(0,), shape=(BLOCK_C,),
                padding_mode=ct.PaddingMode.ZERO),
        ct.float32,
    )
    eps = ct.full(shape=(BLOCK_C,), fill_value=EPS, dtype=ct.float32)
    invstd = ct.rsqrt(var + eps)
    ct.store(invstd_ptr, index=(0,), tile=invstd)


@ct.kernel
def _bn_affine_channels_last_kernel(
    x_ptr,        # bf16 [TOTAL]  (channels-last physical layout: c = offset % C)
    mean_ptr,     # bf16 [C]
    invstd_ptr,   # f32 [C]
    weight_ptr,   # bf16 [C]
    bias_ptr,     # bf16 [C]
    out_ptr,      # bf16 [TOTAL]
    TOTAL: ct.Constant[int],
    C: ct.Constant[int],
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    offsets = pid * BLOCK + ct.arange(BLOCK, dtype=ct.int32)
    valid = offsets < TOTAL

    c = offsets - (offsets // C) * C  # offsets % C

    x = ct.astype(
        ct.load(x_ptr, index=(pid,), shape=(BLOCK,),
                padding_mode=ct.PaddingMode.ZERO),
        ct.float32,
    )
    mean = ct.astype(ct.gather(mean_ptr, (c,)), ct.float32)
    invstd = ct.gather(invstd_ptr, (c,))
    weight = ct.astype(ct.gather(weight_ptr, (c,)), ct.float32)
    bias = ct.astype(ct.gather(bias_ptr, (c,)), ct.float32)

    y = (x - mean) * invstd * weight + bias
    y_bf = ct.astype(y, ct.bfloat16)
    zero_bf = ct.full(shape=(BLOCK,), fill_value=0.0, dtype=ct.bfloat16)
    ct.scatter(out_ptr, (offsets,), ct.where(valid, y_bf, zero_bf), mask=valid)


@ct.kernel
def _bn_affine_contiguous_kernel(
    x_ptr,        # bf16 [TOTAL]  (contiguous: c = (offset // HW) % C)
    mean_ptr,     # bf16 [C]
    invstd_ptr,   # f32 [C]
    weight_ptr,   # bf16 [C]
    bias_ptr,     # bf16 [C]
    out_ptr,      # bf16 [TOTAL]
    TOTAL: ct.Constant[int],
    C: ct.Constant[int],
    HW: ct.Constant[int],
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    offsets = pid * BLOCK + ct.arange(BLOCK, dtype=ct.int32)
    valid = offsets < TOTAL

    c = (offsets // HW) - ((offsets // HW) // C) * C  # (offset // HW) % C

    x = ct.astype(
        ct.load(x_ptr, index=(pid,), shape=(BLOCK,),
                padding_mode=ct.PaddingMode.ZERO),
        ct.float32,
    )
    mean = ct.astype(ct.gather(mean_ptr, (c,)), ct.float32)
    invstd = ct.gather(invstd_ptr, (c,))
    weight = ct.astype(ct.gather(weight_ptr, (c,)), ct.float32)
    bias = ct.astype(ct.gather(bias_ptr, (c,)), ct.float32)

    y = (x - mean) * invstd * weight + bias
    y_bf = ct.astype(y, ct.bfloat16)
    zero_bf = ct.full(shape=(BLOCK,), fill_value=0.0, dtype=ct.bfloat16)
    ct.scatter(out_ptr, (offsets,), ct.where(valid, y_bf, zero_bf), mask=valid)


def _next_pow2(n):
    p = 1
    while p < n:
        p *= 2
    return p


_POINTS = [
    "c6312c18", "1a06d794", "e9bc5ee0", "701aa9ad", "5540c28a", "4c825568",
    "4280fd00", "654b8cd3", "a118ca10", "a047c97b", "76c62ba9", "0a96753f",
    "4c724370", "ef49f65d", "c68fd75d", "ecb9a0d7", "3ddafcba", "3350df9a",
    "e987d149", "1fca60d1", "8e74e392", "b72bb793", "def42dcc", "d6d99242",
    "9f949812", "6129a191", "5959a6c3", "bd03f6f7", "6349ba76", "e2542415",
    "d132f392", "4552b8d5", "40a7d00c", "5e9ad3e0", "4f4eacf9", "7395f576",
    "908bdc99", "d5e26824", "05028dd8", "261d5bdd", "aaa0fce0", "ced04216",
    "d773a707", "d820ae29", "fa15eadf", "6e30cacc", "017f27d2", "5c1756a4",
    "13e72620", "f3036c42", "9a05b224", "0ba75884",
]


def _impl(inputs):
    mean, x, var, weight, bias = inputs
    n, c, h, w = x.shape
    total = n * c * h * w
    hw = h * w
    channels_last = x.stride(1) == 1
    BLOCK = 1024
    C_PAD = _next_pow2(c)

    invstd = torch.empty((c,), device=x.device, dtype=torch.float32)
    out = torch.empty_strided(tuple(x.shape), tuple(x.stride()),
                              device=x.device, dtype=torch.bfloat16)

    stream = torch.cuda.current_stream()
    ct.launch(stream, (1, 1, 1), _invstd_kernel, (var, invstd, c, C_PAD))

    # Flat 1D view of the physical storage (contiguous NCHW or channels-last).
    x_flat = torch.as_strided(x, (total,), (1,))
    out_flat = torch.as_strided(out, (total,), (1,))

    grid = (ct.cdiv(total, BLOCK), 1, 1)
    if channels_last:
        ct.launch(stream, grid, _bn_affine_channels_last_kernel,
                  (x_flat, mean, invstd, weight, bias, out_flat, total, c, BLOCK))
    else:
        ct.launch(stream, grid, _bn_affine_contiguous_kernel,
                  (x_flat, mean, invstd, weight, bias, out_flat, total, c, hw, BLOCK))
    return out


def _make():
    fn = _impl
    for point in _POINTS:
        fn = oracle_impl(hardware="B200", point=point)(fn)
    return fn


oracle_forward = _make()
