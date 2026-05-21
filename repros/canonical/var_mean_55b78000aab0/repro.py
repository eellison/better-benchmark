"""
Standalone repro captured via capture_hook.
Label: torchbench_pytorch_CycleGAN_and_pix2pix_infer
Pattern hash: 55b78000aab0
Shape hash: 15254c9a
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([1, 256, 64, 64], f32), T([1, 256, 64, 64], f32))"

class Repro(torch.nn.Module):
    def forward(self, convolution_20: "f32[1, 256, 64, 64]", add_26: "f32[1, 256, 64, 64]"):
        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pytorch_CycleGAN_and_pix2pix/models/networks.py:564 in forward, code: out = x + self.conv_block(x)  # add skip connections
        var_mean_correction = torch.ops.aten.var_mean.correction(convolution_20, [0, 2, 3], correction = 0, keepdim = True)
        getitem: "f32[1, 256, 1, 1]" = var_mean_correction[0]
        getitem_1: "f32[1, 256, 1, 1]" = var_mean_correction[1];  var_mean_correction = None
        sub_tensor: "f32[1, 256, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_20, getitem_1);  convolution_20 = getitem_1 = None
        add_tensor: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        mul_tensor: "f32[1, 256, 64, 64]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        add_tensor_1: "f32[1, 256, 64, 64]" = torch.ops.aten.add.Tensor(add_26, mul_tensor);  add_26 = mul_tensor = None
        return add_tensor_1



def _default_make_inputs():
    from repro_harness import parse_shapes_config
    return parse_shapes_config(_shapes_config)


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
