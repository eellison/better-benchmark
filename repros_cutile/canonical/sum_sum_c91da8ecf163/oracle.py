"""cuTile port of sum_sum_c91da8ecf163: EfficientNet BN-affine SiLU gate-backward.

Outputs:
  * affine_out: f32 [N, C, H, W]  — bf16-rounded BN affine result cast back to f32.
  * sigmoid_out: bf16 [N, C, 1, 1]  — sigmoid(arg6).
  * gate_grad_out: bf16 [N, C, 1, 1]  — silu_grad-scaled spatial sum.
  * channel_out: f32 [C]  — bf16-then-f32-cast per-channel sum of gate_grad_out.

Hybrid plan:
  * Kernel 1 (_affine_kernel): channels-last flat pointwise producer:
      affine = (x - mean) * invstd * weight + bias   (f32)
      affine_bf16 = ct.astype(affine, bf16)
      affine_out[k] = ct.astype(affine_bf16, f32)   (bf16-round trip)
      silu[k]  = ct.astype(affine / (exp(-affine) + 1), bf16)  (bf16)
      product[k] = ct.astype(grad * silu, bf16)
    Store affine_out (f32) and product (bf16) for downstream torch reductions.
  * torch: sum(product, dim=[2,3], dtype=f32) -> bf16 -> f32 -> sigmoid grad ->
    bf16 gate_grad_out. Then sum(gate_grad_out, dim=[0,2,3]) -> f32.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _affine_silu_prod_kernel(
    x_ptr,           # bf16 [TOTAL] flat NHWC
    grad_ptr,        # bf16 [TOTAL] flat NHWC
    mean_ptr,        # f32 [TOTAL] broadcast
    invstd_ptr,      # f32 [TOTAL] broadcast
    weight_ptr,      # f32 [TOTAL] broadcast
    bias_ptr,        # f32 [TOTAL] broadcast
    affine_out_ptr,  # f32 [TOTAL]
    product_out_ptr, # bf16 [TOTAL]
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    x_bf = ct.load(x_ptr, index=(pid,), shape=(BLOCK,))
    grad_bf = ct.load(grad_ptr, index=(pid,), shape=(BLOCK,))
    mean_v = ct.load(mean_ptr, index=(pid,), shape=(BLOCK,))
    invstd_v = ct.load(invstd_ptr, index=(pid,), shape=(BLOCK,))
    weight_v = ct.load(weight_ptr, index=(pid,), shape=(BLOCK,))
    bias_v = ct.load(bias_ptr, index=(pid,), shape=(BLOCK,))

    x_f = ct.astype(x_bf, ct.float32)
    centered = x_f - mean_v
    normalized = centered * invstd_v
    scaled = normalized * weight_v
    affine_f = scaled + bias_v
    affine_bf = ct.astype(affine_f, ct.bfloat16)
    affine_f32_out = ct.astype(affine_bf, ct.float32)
    ct.store(affine_out_ptr, index=(pid,), tile=affine_f32_out)

    # silu = affine / (exp(-affine) + 1) computed in f32 using the bf16-rounded
    # affine value (matches the eager graph: bf16 cast, then f32 cast, then
    # exp/div, then bf16).
    exp_neg = ct.exp(-affine_f32_out)
    silu_f = affine_f32_out / (exp_neg + 1.0)
    silu_bf = ct.astype(silu_f, ct.bfloat16)
    silu_f32 = ct.astype(silu_bf, ct.float32)
    grad_f = ct.astype(grad_bf, ct.float32)
    product_f = grad_f * silu_f32
    product_bf = ct.astype(product_f, ct.bfloat16)
    ct.store(product_out_ptr, index=(pid,), tile=product_bf)


@oracle_impl(hardware="B200", point="eeebf109")
@oracle_impl(hardware="B200", point="e7f464f6")
@oracle_impl(hardware="B200", point="8b89e909")
@oracle_impl(hardware="B200", point="bb3aa93a")
@oracle_impl(hardware="B200", point="e177ef6b")
@oracle_impl(hardware="B200", point="693f7c58")
@oracle_impl(hardware="B200", point="d58930d9")
@oracle_impl(hardware="B200", point="b8f8666d")
@oracle_impl(hardware="B200", point="dae1d1dd")
@oracle_impl(hardware="B200", point="b90cb54e")
def oracle_forward(inputs):
    x, mean, invstd, weight, bias, grad, gate = inputs
    device = x.device
    n, c, h, w = int(x.shape[0]), int(x.shape[1]), int(x.shape[2]), int(x.shape[3])
    total = n * c * h * w

    # Broadcast per-channel scalars to full-tensor channels-last layout to
    # avoid per-element gather.
    def _bc(vec_c):
        return vec_c.view(1, c, 1, 1).expand(n, c, h, w).contiguous(
            memory_format=torch.channels_last
        )

    # arg1, arg2 shape [1, C, 1, 1] -> broadcast target
    mean_bc = _bc(mean.view(c))
    invstd_bc = _bc(invstd.view(c))
    weight_bc = _bc(weight)
    bias_bc = _bc(bias)

    x_flat = torch.as_strided(x, (total,), (1,))
    grad_flat = torch.as_strided(grad, (total,), (1,))
    mean_flat = torch.as_strided(mean_bc, (total,), (1,))
    invstd_flat = torch.as_strided(invstd_bc, (total,), (1,))
    weight_flat = torch.as_strided(weight_bc, (total,), (1,))
    bias_flat = torch.as_strided(bias_bc, (total,), (1,))

    # affine_out uses channels-last layout matching input.
    affine_flat = torch.empty(total, device=device, dtype=torch.float32)
    product_flat = torch.empty(total, device=device, dtype=torch.bfloat16)

    BLOCK = 256
    stream = torch.cuda.current_stream()
    ct.launch(
        stream, ((total + BLOCK - 1) // BLOCK, 1, 1), _affine_silu_prod_kernel,
        (x_flat, grad_flat, mean_flat, invstd_flat, weight_flat, bias_flat,
         affine_flat, product_flat, BLOCK),
    )

    # Reinterpret flat outputs as channels-last (NCHW) tensors.
    affine_out = torch.as_strided(
        affine_flat, (n, c, h, w), (c * h * w, 1, w * c, c)
    )
    product = torch.as_strided(
        product_flat, (n, c, h, w), (c * h * w, 1, w * c, c)
    )

    # Torch: spatial sum -> bf16 -> f32 -> sigmoid_grad -> bf16.
    sigmoid_gate = torch.sigmoid(gate)             # bf16 [N, C, 1, 1]
    sum1 = product.sum(dim=[2, 3], keepdim=True, dtype=torch.float32)  # f32 [N,C,1,1]
    sum1_bf = sum1.to(torch.bfloat16)
    sum1_f32 = sum1_bf.to(torch.float32)
    sigmoid_f32 = sigmoid_gate.to(torch.float32)
    one_minus = 1 - sigmoid_f32
    mul3 = sigmoid_f32 * one_minus
    mul4 = sum1_f32 * mul3
    gate_grad = mul4.to(torch.bfloat16)             # [N, C, 1, 1]

    channel_out = gate_grad.sum(dim=[0, 2, 3]).to(torch.float32)  # f32 [C]

    return affine_out, sigmoid_gate, gate_grad, channel_out
