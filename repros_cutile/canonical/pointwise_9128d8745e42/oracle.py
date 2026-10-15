"""cuTile port of pointwise_9128d8745e42: RepVGG triple BN-inference + ReLU.

Ports the two-kernel Triton flow (per-branch BN affine then triple add + ReLU)
into two cuTile kernels. cuTile's fp32 arithmetic is already round-to-nearest,
so the Triton `add.rn.f32` / `mul.rn.f32` / `sub.rn.f32` / `sqrt.rn.f32` /
`div.rn.f32` PTX helpers are just Python operators here. Channels are 96/192/
384 (non-pow2); we work in flat NHWC layout with `flat_idx % C` gather of the
per-channel params.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


EPS = 1.0e-5


@ct.kernel
def _bn_branch_flat_kernel(
    mean_ptr,     # bf16[C]
    x_ptr,        # bf16[NUMEL] (flat NHWC)
    var_ptr,      # bf16[C]
    weight_ptr,   # bf16[C]
    bias_ptr,     # bf16[C]
    out_ptr,      # bf16[NUMEL]
    C: ct.Constant[int],
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    idx = pid * BLOCK + ct.arange(BLOCK, dtype=ct.int32)
    c_idx = idx % C

    x = ct.load(x_ptr, index=(pid,), shape=(BLOCK,))
    mean = ct.astype(ct.gather(mean_ptr, c_idx), ct.float32)
    var = ct.astype(ct.gather(var_ptr, c_idx), ct.float32)
    weight = ct.astype(ct.gather(weight_ptr, c_idx), ct.float32)
    bias = ct.astype(ct.gather(bias_ptr, c_idx), ct.float32)

    inv = 1.0 / ct.sqrt(var + EPS)
    normalized = (ct.astype(x, ct.float32) - mean) * inv * weight + bias
    out = ct.astype(normalized, ct.bfloat16)
    ct.store(out_ptr, index=(pid,), tile=out)


@ct.kernel
def _bf16_add3_relu_flat_kernel(
    branch0_ptr,   # bf16[NUMEL]
    branch1_ptr,   # bf16[NUMEL]
    branch2_ptr,   # bf16[NUMEL]
    out_ptr,       # bf16[NUMEL]
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    b0 = ct.load(branch0_ptr, index=(pid,), shape=(BLOCK,))
    b1 = ct.load(branch1_ptr, index=(pid,), shape=(BLOCK,))
    b2 = ct.load(branch2_ptr, index=(pid,), shape=(BLOCK,))
    add01 = ct.astype(ct.astype(b0, ct.float32) + ct.astype(b1, ct.float32), ct.bfloat16)
    add012 = ct.astype(ct.astype(add01, ct.float32) + ct.astype(b2, ct.float32), ct.bfloat16)
    # Triton uses `where(x != x, x, maximum(x, 0.0))` — a NaN-preserving relu.
    # cuTile: ct.where(ct.isnan(x), x, ct.maximum(x, 0.0)).
    zero = ct.astype(ct.zeros((BLOCK,), dtype=ct.bfloat16), ct.bfloat16)
    relu = ct.where(ct.isnan(add012), add012, ct.maximum(add012, zero))
    ct.store(out_ptr, index=(pid,), tile=relu)


def _launch_all(inputs, *, BLOCK):
    (
        mean0, x0, var0, weight0, bias0,
        mean1, x1, var1, weight1, bias1,
        mean2, x2, var2, weight2, bias2,
    ) = inputs

    n, c, h, w = x0.shape
    numel = int(x0.numel())
    # RepVGG shapes come channels-last: strides `[H*W*C, 1, W*C, C]`.
    # Physical storage is NHWC contiguous, so we can view flat 1D.
    # But strided tensors don't necessarily allow direct `.view(numel)` —
    # use `.reshape(-1)` which will allocate a contiguous copy if needed,
    # OR just pass the storage as a flat array via a permute+reshape.
    def _flat(t):
        # NCHW logically but NHWC physical. Permute to NHWC then reshape.
        return t.permute(0, 2, 3, 1).contiguous().view(numel)

    def _empty_like_nhwc():
        # Match x0's stride: (H*W*C, 1, W*C, C).
        return torch.empty_strided(
            (n, c, h, w),
            (h * w * c, 1, w * c, c),
            device=x0.device,
            dtype=torch.bfloat16,
        )

    x0_flat = _flat(x0)
    x1_flat = _flat(x1)
    x2_flat = _flat(x2)
    b0_flat = torch.empty(numel, device=x0.device, dtype=torch.bfloat16)
    b1_flat = torch.empty(numel, device=x0.device, dtype=torch.bfloat16)
    b2_flat = torch.empty(numel, device=x0.device, dtype=torch.bfloat16)

    stream = torch.cuda.current_stream()
    grid = (ct.cdiv(numel, BLOCK), 1, 1)

    ct.launch(stream, grid, _bn_branch_flat_kernel,
              (mean0, x0_flat, var0, weight0, bias0, b0_flat, c, BLOCK))
    ct.launch(stream, grid, _bn_branch_flat_kernel,
              (mean1, x1_flat, var1, weight1, bias1, b1_flat, c, BLOCK))
    ct.launch(stream, grid, _bn_branch_flat_kernel,
              (mean2, x2_flat, var2, weight2, bias2, b2_flat, c, BLOCK))

    out_flat = torch.empty(numel, device=x0.device, dtype=torch.bfloat16)
    ct.launch(stream, grid, _bf16_add3_relu_flat_kernel,
              (b0_flat, b1_flat, b2_flat, out_flat, BLOCK))

    # Reshape the flat NHWC-ordered result back to NCHW with NHWC strides.
    out = out_flat.view(n, h, w, c).permute(0, 3, 1, 2).contiguous(memory_format=torch.channels_last)
    return out


@oracle_impl(hardware="B200", point="319d51bf", BLOCK=1024)
@oracle_impl(hardware="B200", point="8f8763aa", BLOCK=1024)
@oracle_impl(hardware="B200", point="c9e29114", BLOCK=1024)
def oracle_forward(inputs, *, BLOCK):
    return _launch_all(inputs, BLOCK=BLOCK)
