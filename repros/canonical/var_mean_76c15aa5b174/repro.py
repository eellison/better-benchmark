"""
Standalone repro captured via capture_hook.
Label: hf_BartForCausalLM_infer_000
Pattern hash: 76c15aa5b174
Shape hash: 511fb752
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([8192, 1024], f32), T([8, 1024, 1024], f32), T([1024], f32), T([1024], f32), S([8, 1024, 1024]), S([8192, 1024]), S([8192, 1024]), S([8192, 1024]))"

class Repro(torch.nn.Module):
    def forward(self, addmm_65: "f32[8192, 1024]", add_79: "f32[8, 1024, 1024]", arg179_1: "f32[1024]", arg180_1: "f32[1024]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # No stacktrace found for following nodes
        view_default: "f32[8, 1024, 1024]" = torch.ops.aten.view.default(addmm_65, _shape_param_0);  addmm_65 = _shape_param_0 = None
        add_tensor: "f32[8, 1024, 1024]" = torch.ops.aten.add.Tensor(add_79, view_default);  add_79 = view_default = None
        var_mean_correction = torch.ops.aten.var_mean.correction(add_tensor, [2], correction = 0, keepdim = True)
        getitem: "f32[8, 1024, 1]" = var_mean_correction[0]
        getitem_1: "f32[8, 1024, 1]" = var_mean_correction[1];  var_mean_correction = None
        sub_tensor: "f32[8, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_tensor, getitem_1);  add_tensor = getitem_1 = None
        add_tensor_1: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        mul_tensor: "f32[8, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[8, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor, arg179_1);  mul_tensor = arg179_1 = None
        add_tensor_2: "f32[8, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_tensor_1, arg180_1);  mul_tensor_1 = arg180_1 = None
        view_default_1: "f32[8192, 1024]" = torch.ops.aten.view.default(add_tensor_2, _shape_param_1);  _shape_param_1 = None
        view_default_2: "f32[8192, 1024]" = torch.ops.aten.view.default(add_tensor_2, _shape_param_2);  _shape_param_2 = None
        view_default_3: "f32[8192, 1024]" = torch.ops.aten.view.default(add_tensor_2, _shape_param_3);  add_tensor_2 = _shape_param_3 = None
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
