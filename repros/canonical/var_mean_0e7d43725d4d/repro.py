"""
Standalone repro captured via capture_hook.
Label: timm_timm_vit_base_patch16_siglip_256_train_train_000
Pattern hash: 0e7d43725d4d
Shape hash: 6a057fdf
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([32768, 768], f32), T([128, 256, 768], f32, stride=(196608, 1, 256)), T([768], f32), T([768], f32), S([128, 256, 768]))"

class Repro(torch.nn.Module):
    def forward(self, addmm_47: "f32[32768, 768]", add_80: "f32[128, 256, 768]", arg148_1: "f32[768]", arg149_1: "f32[768]", _shape_param_0):
        # No stacktrace found for following nodes
        view_default: "f32[128, 256, 768]" = torch.ops.aten.view.default(addmm_47, _shape_param_0);  addmm_47 = _shape_param_0 = None
        add_tensor: "f32[128, 256, 768]" = torch.ops.aten.add.Tensor(add_80, view_default);  add_80 = view_default = None
        var_mean_correction = torch.ops.aten.var_mean.correction(add_tensor, [2], correction = 0, keepdim = True)
        getitem: "f32[128, 256, 1]" = var_mean_correction[0]
        getitem_1: "f32[128, 256, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor_1: "f32[128, 256, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-06);  getitem = None
        rsqrt_default: "f32[128, 256, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        sub_tensor: "f32[128, 256, 768]" = torch.ops.aten.sub.Tensor(add_tensor, getitem_1);  add_tensor = getitem_1 = None
        mul_tensor: "f32[128, 256, 768]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = None
        mul_tensor_1: "f32[128, 256, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, arg148_1);  mul_tensor = arg148_1 = None
        add_tensor_2: "f32[128, 256, 768]" = torch.ops.aten.add.Tensor(mul_tensor_1, arg149_1);  mul_tensor_1 = arg149_1 = None
        clone_default: "f32[128, 256, 768]" = torch.ops.aten.clone.default(add_tensor_2, memory_format = torch.contiguous_format);  add_tensor_2 = None
        _unsafe_view_default: "f32[32768, 768]" = torch.ops.aten._unsafe_view.default(clone_default, [32768, 768]);  clone_default = None
        div_tensor: "f32[128, 256, 1]" = torch.ops.aten.div.Tensor(rsqrt_default, 768);  rsqrt_default = None
        return (_unsafe_view_default, div_tensor)

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
