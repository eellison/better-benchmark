"""
Standalone repro captured via capture_hook.
Label: hf_AllenaiLongformerBase_infer_002
Pattern hash: 7d9e5a9d28a4
Shape hash: 8b1c40de
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([8192, 768], f32), T([8, 1024, 768], f32), T([768], f32), T([768], f32), S([8, 1024, 768]), S([8192, 768]), S([8192, 768]), S([8192, 768]))"

class Repro(torch.nn.Module):
    def forward(self, addmm_21: "f32[8192, 768]", add_127: "f32[8, 1024, 768]", arg177_1: "f32[768]", arg178_1: "f32[768]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # No stacktrace found for following nodes
        view_default: "f32[8, 1024, 768]" = torch.ops.aten.view.default(addmm_21, _shape_param_0);  addmm_21 = _shape_param_0 = None
        add_tensor: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(view_default, add_127);  view_default = add_127 = None
        var_mean_correction = torch.ops.aten.var_mean.correction(add_tensor, [2], correction = 0, keepdim = True)
        getitem: "f32[8, 1024, 1]" = var_mean_correction[0]
        getitem_1: "f32[8, 1024, 1]" = var_mean_correction[1];  var_mean_correction = None
        sub_tensor: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(add_tensor, getitem_1);  add_tensor = getitem_1 = None
        add_tensor_1: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        mul_tensor: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, arg177_1);  mul_tensor = arg177_1 = None
        add_tensor_2: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_tensor_1, arg178_1);  mul_tensor_1 = arg178_1 = None
        permute_default: "f32[1024, 8, 768]" = torch.ops.aten.permute.default(add_tensor_2, [1, 0, 2]);  add_tensor_2 = None
        clone_default: "f32[1024, 8, 768]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format)
        view_default_1: "f32[8192, 768]" = torch.ops.aten.view.default(clone_default, _shape_param_1);  clone_default = _shape_param_1 = None
        clone_default_1: "f32[1024, 8, 768]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format)
        view_default_2: "f32[8192, 768]" = torch.ops.aten.view.default(clone_default_1, _shape_param_2);  clone_default_1 = _shape_param_2 = None
        clone_default_2: "f32[1024, 8, 768]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        view_default_3: "f32[8192, 768]" = torch.ops.aten.view.default(clone_default_2, _shape_param_3);  clone_default_2 = _shape_param_3 = None
        return (view_default_1, view_default_2, view_default_3)

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
