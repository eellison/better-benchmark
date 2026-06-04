"""
Standalone repro captured via capture_hook.
Label: timm_convnextv2_nano.fcmae_ft_in22k_in1k_infer_000
Pattern hash: e4d2655a5279
Shape hash: d69d5f08
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([128, 640, 7, 7], f32), T([640], f32), T([640], f32))"

class Repro(torch.nn.Module):
    def forward(self, convolution_43: "f32[128, 640, 7, 7]", arg149_1: "f32[640]", arg150_1: "f32[640]"):
        # No stacktrace found for following nodes
        permute_default: "f32[128, 7, 7, 640]" = torch.ops.aten.permute.default(convolution_43, [0, 2, 3, 1]);  convolution_43 = None
        var_mean_correction = torch.ops.aten.var_mean.correction(permute_default, [3], correction = 0, keepdim = True)
        getitem: "f32[128, 7, 7, 1]" = var_mean_correction[0]
        getitem_1: "f32[128, 7, 7, 1]" = var_mean_correction[1];  var_mean_correction = None
        sub_tensor: "f32[128, 7, 7, 640]" = torch.ops.aten.sub.Tensor(permute_default, getitem_1);  permute_default = getitem_1 = None
        add_tensor: "f32[128, 7, 7, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-06);  getitem = None
        rsqrt_default: "f32[128, 7, 7, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        mul_tensor: "f32[128, 7, 7, 640]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[128, 7, 7, 640]" = torch.ops.aten.mul.Tensor(mul_tensor, arg149_1);  mul_tensor = arg149_1 = None
        add_tensor_1: "f32[128, 7, 7, 640]" = torch.ops.aten.add.Tensor(mul_tensor_1, arg150_1);  mul_tensor_1 = arg150_1 = None
        permute_default_1: "f32[128, 640, 7, 7]" = torch.ops.aten.permute.default(add_tensor_1, [0, 3, 1, 2]);  add_tensor_1 = None
        return permute_default_1

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
