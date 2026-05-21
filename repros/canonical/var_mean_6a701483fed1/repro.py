"""
Standalone repro captured via capture_hook.
Label: timm_timm_vit_base_patch14_dinov2.lvd142m_train_train_000
Pattern hash: 6a701483fed1
Shape hash: eed64cad
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
_shapes_config = "(T([175360, 768], f32), T([768], f32), T([128, 1370, 768], f32), T([768], f32), T([768], f32), S([128, 1370, 768]), S([175360, 768]))"

class Repro(torch.nn.Module):
    def forward(self, addmm_45: "f32[175360, 768]", arg165_1: "f32[768]", add_77: "f32[128, 1370, 768]", arg166_1: "f32[768]", arg167_1: "f32[768]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        view_default: "f32[128, 1370, 768]" = torch.ops.aten.view.default(addmm_45, _shape_param_0);  addmm_45 = _shape_param_0 = None
        mul_tensor: "f32[128, 1370, 768]" = torch.ops.aten.mul.Tensor(view_default, arg165_1);  view_default = arg165_1 = None
        add_tensor: "f32[128, 1370, 768]" = torch.ops.aten.add.Tensor(add_77, mul_tensor);  add_77 = mul_tensor = None
        var_mean_correction = torch.ops.aten.var_mean.correction(add_tensor, [2], correction = 0, keepdim = True)
        getitem: "f32[128, 1370, 1]" = var_mean_correction[0]
        getitem_1: "f32[128, 1370, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor_1: "f32[128, 1370, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-06);  getitem = None
        rsqrt_default: "f32[128, 1370, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        sub_tensor: "f32[128, 1370, 768]" = torch.ops.aten.sub.Tensor(add_tensor, getitem_1);  add_tensor = getitem_1 = None
        mul_tensor_1: "f32[128, 1370, 768]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = None
        mul_tensor_2: "f32[128, 1370, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_1, arg166_1);  mul_tensor_1 = arg166_1 = None
        add_tensor_2: "f32[128, 1370, 768]" = torch.ops.aten.add.Tensor(mul_tensor_2, arg167_1);  mul_tensor_2 = arg167_1 = None
        view_default_1: "f32[175360, 768]" = torch.ops.aten.view.default(add_tensor_2, _shape_param_1);  add_tensor_2 = _shape_param_1 = None
        div_tensor: "f32[128, 1370, 1]" = torch.ops.aten.div.Tensor(rsqrt_default, 768);  rsqrt_default = None
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
