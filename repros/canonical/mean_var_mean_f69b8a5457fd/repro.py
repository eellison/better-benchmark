"""
Standalone repro captured via capture_hook.
Label: timm_beit_base_patch16_224_train_000
Pattern hash: f69b8a5457fd
Shape hash: 3eea3606
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([25216, 768], f32), T([768], f32), T([128, 197, 768], f32), T([768], f32), T([768], f32), S([128, 197, 768]))"

class Repro(torch.nn.Module):
    def forward(self, addmm_47: "f32[25216, 768]", arg213_1: "f32[768]", add_79: "f32[128, 197, 768]", arg220_1: "f32[768]", arg221_1: "f32[768]", _shape_param_0):
        # No stacktrace found for following nodes
        view_default: "f32[128, 197, 768]" = torch.ops.aten.view.default(addmm_47, _shape_param_0);  addmm_47 = _shape_param_0 = None
        mul_tensor: "f32[128, 197, 768]" = torch.ops.aten.mul.Tensor(arg213_1, view_default);  arg213_1 = view_default = None
        add_tensor: "f32[128, 197, 768]" = torch.ops.aten.add.Tensor(add_79, mul_tensor);  add_79 = mul_tensor = None
        slice_tensor: "f32[128, 196, 768]" = torch.ops.aten.slice.Tensor(add_tensor, 1, 1, 9223372036854775807);  add_tensor = None
        mean_dim: "f32[128, 768]" = torch.ops.aten.mean.dim(slice_tensor, [1]);  slice_tensor = None
        var_mean_correction = torch.ops.aten.var_mean.correction(mean_dim, [1], correction = 0, keepdim = True)
        getitem: "f32[128, 1]" = var_mean_correction[0]
        getitem_1: "f32[128, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor_1: "f32[128, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-06);  getitem = None
        rsqrt_default: "f32[128, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        sub_tensor: "f32[128, 768]" = torch.ops.aten.sub.Tensor(mean_dim, getitem_1);  mean_dim = getitem_1 = None
        mul_tensor_1: "f32[128, 768]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = None
        mul_tensor_2: "f32[128, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_1, arg220_1);  mul_tensor_1 = arg220_1 = None
        add_tensor_2: "f32[128, 768]" = torch.ops.aten.add.Tensor(mul_tensor_2, arg221_1);  mul_tensor_2 = arg221_1 = None
        div_tensor: "f32[128, 1]" = torch.ops.aten.div.Tensor(rsqrt_default, 768);  rsqrt_default = None
        return (add_tensor_2, div_tensor)

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
