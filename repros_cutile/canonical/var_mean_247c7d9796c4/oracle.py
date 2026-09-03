"""cuTile port of var_mean_247c7d9796c4: per-channel var_mean + relu + reflect pad.

Var/mean is over (0, 2, 3), normalize per channel, bf16 round, relu, then
reflection pad 1x1x1x1 (spatial). Uses cuTile row-per-channel kernel and torch
for the reflection pad.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


EPS = 1.0e-5


@ct.kernel
def _instance_norm_relu_kernel(
    x_ptr,       # bf16 [C, HW]
    out_ptr,     # bf16 [C, HW]
    HW: ct.Constant[int],
    HW_F: ct.Constant[float],
):
    row = ct.bid(0)  # channel
    x = ct.load(x_ptr, index=(row, 0), shape=(1, HW))
    x_f = ct.astype(x, ct.float32)
    mean_val = ct.sum(x_f) * (1.0 / HW_F)
    centered = x_f - mean_val
    variance_val = ct.sum(centered * centered) * (1.0 / HW_F)
    invstd_val = ct.rsqrt(variance_val + EPS)
    normalized = centered * invstd_val
    norm_bf16 = ct.astype(normalized, ct.bfloat16)
    norm_f = ct.astype(norm_bf16, ct.float32)
    zero = ct.full(shape=(1, HW), fill_value=0.0, dtype=ct.float32)
    relu_val = ct.where(norm_f > zero, norm_f, zero)
    ct.store(out_ptr, index=(row, 0), tile=ct.astype(relu_val, ct.bfloat16))


@oracle_impl(hardware="B200", point="1b4e0bc8")
def oracle_forward(inputs):
    (x,) = inputs
    batch, channels, height, width = x.shape
    hw = height * width
    # Reshape as [C, HW] because batch=1 for this shape point.
    assert batch == 1
    x_2d = x.view(channels, hw)
    device = x.device

    norm_relu = torch.empty_strided(
        (channels, hw), (hw, 1), device=device, dtype=torch.bfloat16,
    )
    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (channels, 1, 1), _instance_norm_relu_kernel,
        (x_2d, norm_relu, hw, float(hw)),
    )
    norm_relu_nchw = norm_relu.view(batch, channels, height, width)
    # Reflection pad by (1, 1, 1, 1)
    padded = torch.nn.functional.pad(norm_relu_nchw.to(torch.float32), [1, 1, 1, 1], mode="reflect")
    return padded.to(torch.bfloat16)
