"""cuTile port of pointwise_598163daa1a2: DeBERTa head clone/div/permute.

Materializes a bf16 [B, H, S, D] contiguous clone from packed [B, S, H, D]
input, dividing by a scalar bf16 factor. Returns [B*H, D, S] (via permute) and
[B*H, S, D].
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


B = 8
S = 512
H = 24
D = 64


@ct.kernel
def _deberta_head_div_kernel(
    src,      # (B, S, H, D)
    scale,    # (1,)
    dst,      # (B, H, S, D)
    BLOCK_D: ct.Constant[int],
):
    b = ct.bid(0)
    h = ct.bid(1)
    s = ct.bid(2)
    values = ct.load(src, index=(b, s, h, 0), shape=(1, 1, 1, BLOCK_D))
    values_f = ct.astype(values, ct.float32)
    scale_v = ct.load(scale, index=(0,), shape=(1,))
    scale_f = ct.astype(scale_v, ct.float32)
    # broadcast scalar
    scale_bc = ct.reshape(scale_f, (1, 1, 1, 1))
    result = values_f / scale_bc
    ct.store(dst, index=(b, h, s, 0), tile=ct.astype(result, ct.bfloat16))


@oracle_impl(hardware="B200", point="6bfc0be7")
def oracle_forward(inputs):
    x, scale, _shape0, _shape1, _shape2 = inputs
    base = torch.empty_strided(
        (192, 512, 64),
        (32768, 64, 1),
        device=x.device,
        dtype=torch.bfloat16,
    )
    # Interpret input as (B, S, H, D); output as (B, H, S, D).
    src4 = x.view(B, S, H, D)
    dst4 = base.view(B, H, S, D)
    scale_v = scale.view(1)
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (B, H, S),
        _deberta_head_div_kernel,
        (src4, scale_v, dst4, D),
    )
    return base.permute(0, 2, 1), base
