"""
Standalone repro captured via capture_hook.
Label: timm_beit_base_patch16_224_train_000
Pattern hash: 0361de9eae81
Shape hash: 3eea3606
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([25216, 768], f32), T([768], f32), T([128, 197, 768], f32), T([768], f32), T([768], f32), S([128, 197, 768]), S([25216, 768]))"

class Repro(torch.nn.Module):
    def forward(self, addmm_45: "f32[25216, 768]", arg202_1: "f32[768]", add_76: "f32[128, 197, 768]", arg214_1: "f32[768]", arg215_1: "f32[768]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        view_default: "f32[128, 197, 768]" = torch.ops.aten.view.default(addmm_45, _shape_param_0);  addmm_45 = _shape_param_0 = None
        mul_tensor: "f32[128, 197, 768]" = torch.ops.aten.mul.Tensor(arg202_1, view_default);  arg202_1 = view_default = None
        add_tensor: "f32[128, 197, 768]" = torch.ops.aten.add.Tensor(add_76, mul_tensor);  add_76 = mul_tensor = None
        var_mean_correction = torch.ops.aten.var_mean.correction(add_tensor, [2], correction = 0, keepdim = True)
        getitem: "f32[128, 197, 1]" = var_mean_correction[0]
        getitem_1: "f32[128, 197, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor_1: "f32[128, 197, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-06);  getitem = None
        rsqrt_default: "f32[128, 197, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        sub_tensor: "f32[128, 197, 768]" = torch.ops.aten.sub.Tensor(add_tensor, getitem_1);  add_tensor = getitem_1 = None
        mul_tensor_1: "f32[128, 197, 768]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = None
        mul_tensor_2: "f32[128, 197, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_1, arg214_1);  mul_tensor_1 = arg214_1 = None
        add_tensor_2: "f32[128, 197, 768]" = torch.ops.aten.add.Tensor(mul_tensor_2, arg215_1);  mul_tensor_2 = arg215_1 = None
        view_default_1: "f32[25216, 768]" = torch.ops.aten.view.default(add_tensor_2, _shape_param_1);  add_tensor_2 = _shape_param_1 = None
        div_tensor: "f32[128, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_default, 768);  rsqrt_default = None
        return (view_default_1, div_tensor)

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
