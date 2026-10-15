"""cuTile port of pointwise_ada23cf7aec6 (SCHEDULER_FUSION): DenseNet 2x2
avg_pool + BN inference + NaN-preserving ReLU.

Uses torch for the avg_pool2d (which is what the eager Repro does — no bf16
rounding boundary needed since the eager reference stores the bf16 avg_pool
output directly). Then the BN inference affine + ReLU is a per-element
elementwise pass over the pooled tensor, ported to cuTile via `ct.gather` for
per-channel params.
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
def _bn_relu_contiguous_kernel(
    x_ptr,        # bf16 [TOTAL]  (contiguous NCHW: c = (offset // HW) % C)
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
    # NaN-preserving ReLU: keep NaN; otherwise max(x, 0)
    is_nan = y_bf != y_bf
    relu = ct.where((y_bf > zero_bf) | is_nan, y_bf, zero_bf)
    ct.scatter(out_ptr, (offsets,), ct.where(valid, relu, zero_bf), mask=valid)


def _next_pow2(n):
    p = 1
    while p < n:
        p *= 2
    return p


@oracle_impl(hardware="B200", point="793856d7")
@oracle_impl(hardware="B200", point="66d30643")
@oracle_impl(hardware="B200", point="3a0fa905")
@oracle_impl(hardware="B200", point="8d5f7d9c")
@oracle_impl(hardware="B200", point="31290814")
@oracle_impl(hardware="B200", point="ec03f8a7")
def oracle_forward(inputs, **_kwargs):
    x, mean, var, weight, bias = inputs
    # avg_pool2d 2x2 stride-2
    pooled = torch.ops.aten.avg_pool2d.default(x, [2, 2], [2, 2])
    pooled_contig = pooled.contiguous()

    n, c, oh, ow = pooled_contig.shape
    total = pooled_contig.numel()
    hw = oh * ow
    BLOCK = 1024
    C_PAD = _next_pow2(c)

    invstd = torch.empty((c,), device=x.device, dtype=torch.float32)
    relu = torch.empty_strided(
        tuple(pooled_contig.shape), tuple(pooled_contig.stride()),
        device=x.device, dtype=torch.bfloat16,
    )

    stream = torch.cuda.current_stream()
    ct.launch(stream, (1, 1, 1), _invstd_kernel, (var, invstd, c, C_PAD))

    pooled_flat = torch.as_strided(pooled_contig, (total,), (1,))
    relu_flat = torch.as_strided(relu, (total,), (1,))

    grid = (ct.cdiv(total, BLOCK), 1, 1)
    ct.launch(stream, grid, _bn_relu_contiguous_kernel,
              (pooled_flat, mean, invstd, weight, bias, relu_flat,
               total, c, hw, BLOCK))
    return pooled, relu
