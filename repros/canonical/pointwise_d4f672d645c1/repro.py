"""
Standalone repro captured via capture_hook.
Label: timm_nfnet_l0_infer_000
Pattern hash: d4f672d645c1
Shape hash: 2e9b11f5
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
_shapes_config = "(T([128, 1536, 1, 1], f32), T([128, 1536, 14, 14], f32), T([128, 1536, 14, 14], f32))"

class Repro(torch.nn.Module):
    def forward(self, convolution_60: "f32[128, 1536, 1, 1]", convolution_58: "f32[128, 1536, 14, 14]", add_81: "f32[128, 1536, 14, 14]"):
        # No stacktrace found for following nodes
        sigmoid_default: "f32[128, 1536, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_60);  convolution_60 = None
        mul_tensor: "f32[128, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_58, sigmoid_default);  convolution_58 = sigmoid_default = None
        mul_tensor_1: "f32[128, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_tensor, 2.0);  mul_tensor = None
        mul_tensor_2: "f32[128, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_tensor_1, 0.2);  mul_tensor_1 = None
        add_tensor: "f32[128, 1536, 14, 14]" = torch.ops.aten.add.Tensor(mul_tensor_2, add_81);  mul_tensor_2 = add_81 = None
        neg_default: "f32[128, 1536, 14, 14]" = torch.ops.aten.neg.default(add_tensor)
        exp_default: "f32[128, 1536, 14, 14]" = torch.ops.aten.exp.default(neg_default);  neg_default = None
        add_tensor_1: "f32[128, 1536, 14, 14]" = torch.ops.aten.add.Tensor(exp_default, 1);  exp_default = None
        div_tensor: "f32[128, 1536, 14, 14]" = torch.ops.aten.div.Tensor(add_tensor, add_tensor_1);  add_tensor = add_tensor_1 = None
        mul_tensor_3: "f32[128, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(div_tensor, 0.8980265101338745);  div_tensor = None
        avg_pool2d_default: "f32[128, 1536, 7, 7]" = torch.ops.aten.avg_pool2d.default(mul_tensor_3, [2, 2], [2, 2], [0, 0], True, False);  mul_tensor_3 = None
        return avg_pool2d_default



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
