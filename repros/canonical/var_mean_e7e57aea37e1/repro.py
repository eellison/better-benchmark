"""
Standalone repro captured via capture_hook.
Label: torchbench_pytorch_CycleGAN_and_pix2pix_infer_000
Pattern hash: e7e57aea37e1
Shape hash: 6a26adea
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
_shapes_config = "(T([1, 128, 128, 128], f32))"

class Repro(torch.nn.Module):
    def forward(self, convolution_21: "f32[1, 128, 128, 128]"):
        # No stacktrace found for following nodes
        var_mean_correction = torch.ops.aten.var_mean.correction(convolution_21, [0, 2, 3], correction = 0, keepdim = True)
        getitem: "f32[1, 128, 1, 1]" = var_mean_correction[0]
        getitem_1: "f32[1, 128, 1, 1]" = var_mean_correction[1];  var_mean_correction = None
        sub_tensor: "f32[1, 128, 128, 128]" = torch.ops.aten.sub.Tensor(convolution_21, getitem_1);  convolution_21 = getitem_1 = None
        add_tensor: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        mul_tensor: "f32[1, 128, 128, 128]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        relu_default: "f32[1, 128, 128, 128]" = torch.ops.aten.relu.default(mul_tensor);  mul_tensor = None
        return relu_default



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
