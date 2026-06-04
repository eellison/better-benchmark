"""
Standalone repro captured via capture_hook.
Label: timm_mobilevit_s_train_000
Pattern hash: b95d34ef7580
Shape hash: 010de0a7
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([8192, 240], f32), T([512, 16, 240], f32), T([240], f32), T([240], f32), S([512, 16, 240]), S([128, 4, 16, -1]), S([122880, 4, 2, 2]), S([128, 240, 8, 8]))"

class Repro(torch.nn.Module):
    def forward(self, addmm_35: "f32[8192, 240]", add_231: "f32[512, 16, 240]", arg290_1: "f32[240]", arg291_1: "f32[240]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # No stacktrace found for following nodes
        view_default: "f32[512, 16, 240]" = torch.ops.aten.view.default(addmm_35, _shape_param_0);  addmm_35 = _shape_param_0 = None
        add_tensor: "f32[512, 16, 240]" = torch.ops.aten.add.Tensor(add_231, view_default);  add_231 = view_default = None
        var_mean_correction = torch.ops.aten.var_mean.correction(add_tensor, [2], correction = 0, keepdim = True)
        getitem: "f32[512, 16, 1]" = var_mean_correction[0]
        getitem_1: "f32[512, 16, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor_1: "f32[512, 16, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[512, 16, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        sub_tensor: "f32[512, 16, 240]" = torch.ops.aten.sub.Tensor(add_tensor, getitem_1);  add_tensor = getitem_1 = None
        mul_tensor: "f32[512, 16, 240]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = None
        mul_tensor_1: "f32[512, 16, 240]" = torch.ops.aten.mul.Tensor(mul_tensor, arg290_1);  mul_tensor = arg290_1 = None
        add_tensor_2: "f32[512, 16, 240]" = torch.ops.aten.add.Tensor(mul_tensor_1, arg291_1);  mul_tensor_1 = arg291_1 = None
        view_default_1: "f32[128, 4, 16, 240]" = torch.ops.aten.view.default(add_tensor_2, _shape_param_1);  add_tensor_2 = _shape_param_1 = None
        permute_default: "f32[128, 240, 16, 4]" = torch.ops.aten.permute.default(view_default_1, [0, 3, 2, 1]);  view_default_1 = None
        clone_default: "f32[128, 240, 16, 4]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        view_default_2: "f32[122880, 4, 2, 2]" = torch.ops.aten.view.default(clone_default, _shape_param_2);  clone_default = _shape_param_2 = None
        permute_default_1: "f32[122880, 2, 4, 2]" = torch.ops.aten.permute.default(view_default_2, [0, 2, 1, 3]);  view_default_2 = None
        clone_default_1: "f32[122880, 2, 4, 2]" = torch.ops.aten.clone.default(permute_default_1, memory_format = torch.contiguous_format);  permute_default_1 = None
        view_default_3: "f32[128, 240, 8, 8]" = torch.ops.aten.view.default(clone_default_1, _shape_param_3);  clone_default_1 = _shape_param_3 = None
        div_tensor: "f32[512, 16, 1]" = torch.ops.aten.div.Tensor(rsqrt_default, 240);  rsqrt_default = None
        return (view_default_3, div_tensor)

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
