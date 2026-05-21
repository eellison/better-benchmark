"""
Standalone repro captured via capture_hook.
Label: timm_dm_nfnet_f0_infer_000
Pattern hash: d29b1b2a009e
Shape hash: 102f3a82
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
_shapes_config = "(T([128, 1536, 1, 1], f32), T([128, 1536, 6, 6], f32), T([], f32), T([128, 1536, 6, 6], f32))"

class Repro(torch.nn.Module):
    def forward(self, convolution_73: "f32[128, 1536, 1, 1]", convolution_71: "f32[128, 1536, 6, 6]", arg211_1: "f32[]", add_100: "f32[128, 1536, 6, 6]"):
        # No stacktrace found for following nodes
        sigmoid_default: "f32[128, 1536, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_73);  convolution_73 = None
        mul_tensor: "f32[128, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(convolution_71, sigmoid_default);  convolution_71 = sigmoid_default = None
        mul_tensor_1: "f32[128, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(mul_tensor, 2.0);  mul_tensor = None
        mul_tensor_2: "f32[128, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(mul_tensor_1, arg211_1);  mul_tensor_1 = arg211_1 = None
        mul_tensor_3: "f32[128, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(mul_tensor_2, 0.2);  mul_tensor_2 = None
        add_tensor: "f32[128, 1536, 6, 6]" = torch.ops.aten.add.Tensor(mul_tensor_3, add_100);  mul_tensor_3 = add_100 = None
        mul_tensor_4: "f32[128, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(add_tensor, 0.5)
        mul_tensor_5: "f32[128, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(add_tensor, 0.7071067811865476);  add_tensor = None
        erf_default: "f32[128, 1536, 6, 6]" = torch.ops.aten.erf.default(mul_tensor_5);  mul_tensor_5 = None
        add_tensor_1: "f32[128, 1536, 6, 6]" = torch.ops.aten.add.Tensor(erf_default, 1);  erf_default = None
        mul_tensor_6: "f32[128, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(mul_tensor_4, add_tensor_1);  mul_tensor_4 = add_tensor_1 = None
        mul_tensor_7: "f32[128, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(mul_tensor_6, 1.7015043497085571);  mul_tensor_6 = None
        mul_tensor_8: "f32[128, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(mul_tensor_7, 0.9622504486493761);  mul_tensor_7 = None
        return mul_tensor_8



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
