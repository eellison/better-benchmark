"""
Standalone repro captured via capture_hook.
Label: hf_GoogleFnet_train_000
Pattern hash: 9010cd062526
Shape hash: fcd58081
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([32, 512, 768, 2], f32), T([32, 512, 768], f32), T([768], f32), T([768], f32), S([16384, 768]))"

class Repro(torch.nn.Module):
    def forward(self, view_as_real_11: "f32[32, 512, 768, 2]", add_91: "f32[32, 512, 768]", arg98_1: "f32[768]", arg99_1: "f32[768]", _shape_param_0):
        # No stacktrace found for following nodes
        select_int: "f32[32, 512, 768]" = torch.ops.aten.select.int(view_as_real_11, 3, 0);  view_as_real_11 = None
        add_tensor: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_91, select_int);  add_91 = select_int = None
        var_mean_correction = torch.ops.aten.var_mean.correction(add_tensor, [2], correction = 0, keepdim = True)
        getitem: "f32[32, 512, 1]" = var_mean_correction[0]
        getitem_1: "f32[32, 512, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor_1: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
        rsqrt_default: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        sub_tensor: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_tensor, getitem_1);  add_tensor = getitem_1 = None
        mul_tensor: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = None
        mul_tensor_1: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, arg98_1);  mul_tensor = arg98_1 = None
        add_tensor_2: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_tensor_1, arg99_1);  mul_tensor_1 = arg99_1 = None
        view_default: "f32[16384, 768]" = torch.ops.aten.view.default(add_tensor_2, _shape_param_0);  add_tensor_2 = _shape_param_0 = None
        div_tensor: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_default, 768);  rsqrt_default = None
        return (view_default, div_tensor)

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
