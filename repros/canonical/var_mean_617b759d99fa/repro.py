"""
Standalone repro captured via capture_hook.
Label: timm_convnextv2_nano.fcmae_ft_in22k_in1k_infer_000
Pattern hash: 617b759d99fa
Shape hash: 838bcbcb
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([128, 320, 14, 14], f32), T([128, 320, 14, 14], f32), T([320], f32), T([320], f32))"

class Repro(torch.nn.Module):
    def forward(self, convolution_38: "f32[128, 320, 14, 14]", add_71: "f32[128, 320, 14, 14]", arg133_1: "f32[320]", arg134_1: "f32[320]"):
        # No stacktrace found for following nodes
        add_tensor: "f32[128, 320, 14, 14]" = torch.ops.aten.add.Tensor(convolution_38, add_71);  convolution_38 = add_71 = None
        permute_default: "f32[128, 14, 14, 320]" = torch.ops.aten.permute.default(add_tensor, [0, 2, 3, 1]);  add_tensor = None
        var_mean_correction = torch.ops.aten.var_mean.correction(permute_default, [3], correction = 0, keepdim = True)
        getitem: "f32[128, 14, 14, 1]" = var_mean_correction[0]
        getitem_1: "f32[128, 14, 14, 1]" = var_mean_correction[1];  var_mean_correction = None
        sub_tensor: "f32[128, 14, 14, 320]" = torch.ops.aten.sub.Tensor(permute_default, getitem_1);  permute_default = getitem_1 = None
        add_tensor_1: "f32[128, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-06);  getitem = None
        rsqrt_default: "f32[128, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        mul_tensor: "f32[128, 14, 14, 320]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[128, 14, 14, 320]" = torch.ops.aten.mul.Tensor(mul_tensor, arg133_1);  mul_tensor = arg133_1 = None
        add_tensor_2: "f32[128, 14, 14, 320]" = torch.ops.aten.add.Tensor(mul_tensor_1, arg134_1);  mul_tensor_1 = arg134_1 = None
        permute_default_1: "f32[128, 320, 14, 14]" = torch.ops.aten.permute.default(add_tensor_2, [0, 3, 1, 2]);  add_tensor_2 = None
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
