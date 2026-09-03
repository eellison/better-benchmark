"""cuTile port of pointwise_1df91f562302: NFNet exact-erf GELU scale bf16.

Ports the Triton `_nfnet_gelu_scale_bf16_kernel` to cuda.tile. cuTile does not
provide `erf`, so we substitute an Abramowitz-Stegun (7.1.26) 5-coefficient
polynomial approximation; on the input range checked here, the maximum absolute
error against `math.erf` is ~3e-7, which vanishes after the bf16 round trip.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _nfnet_gelu_scale_bf16_kernel(
    input_ptr,
    output_ptr,
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    x_bf16 = ct.load(input_ptr, index=(pid,), shape=(BLOCK,))
    x = ct.astype(x_bf16, ct.float32)
    # erf(z) via Abramowitz-Stegun 7.1.26:
    #   sign(z) * (1 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1) * t * exp(-z*z))
    #   where t = 1 / (1 + p*|z|)
    z = x * 0.7071067811865476
    sign_z = ct.where(z < 0.0, ct.full(shape=(BLOCK,), fill_value=-1.0, dtype=ct.float32),
                              ct.full(shape=(BLOCK,), fill_value=1.0, dtype=ct.float32))
    az = ct.where(z < 0.0, -z, z)
    p = 0.3275911
    t = 1.0 / (1.0 + p * az)
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    poly = ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t
    erf_z = sign_z * (1.0 - poly * ct.exp(-az * az))
    gelu = (x * 0.5) * (erf_z + 1.0)
    gelu_bf16 = ct.astype(gelu, ct.bfloat16)
    scaled = ct.astype(gelu_bf16, ct.float32) * 1.7015043497085571
    ct.store(output_ptr, index=(pid,), tile=ct.astype(scaled, ct.bfloat16))


@oracle_impl(hardware="B200", point="4c47a3fc")
@oracle_impl(hardware="B200", point="192c8e99")
@oracle_impl(hardware="B200", point="b00342cf")
@oracle_impl(hardware="B200", point="f77e434f")
@oracle_impl(hardware="B200", point="209c35c7")
@oracle_impl(hardware="B200", point="0630f81f")
@oracle_impl(hardware="B200", point="926b386f")
@oracle_impl(hardware="B200", point="95428590")
@oracle_impl(hardware="B200", point="b358c4cf")
@oracle_impl(hardware="B200", point="1acf97f6")
@oracle_impl(hardware="B200", point="6fef8a13")
@oracle_impl(hardware="B200", point="20ba0bab")
def oracle_forward(inputs):
    (x,) = inputs
    out = torch.empty_strided(
        tuple(x.shape),
        tuple(x.stride()),
        device=x.device,
        dtype=torch.bfloat16,
    )
    n_elements = x.numel()
    BLOCK = 1024
    # The input tensor is channels-last (non-contiguous). Since the operation
    # is fully elementwise, we can safely use as_strided on the physical
    # storage as a flat 1D view of length numel.
    x_flat = torch.as_strided(x, (n_elements,), (1,))
    out_flat = torch.as_strided(out, (n_elements,), (1,))
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(n_elements, BLOCK), 1, 1),
        _nfnet_gelu_scale_bf16_kernel,
        (x_flat, out_flat, BLOCK),
    )
    return out
