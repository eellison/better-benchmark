"""
Standalone repro captured via capture_hook.
Label: hf_M2M100ForConditionalGeneration_infer_000
Pattern hash: aabf4b70dc64
Shape hash: 11f07fbc
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([8192, 1024], f32), T([64, 128, 1024], f32), T([1024], f32), T([1024], f32), S([64, 128, 1024]), S([8192, 1024]), S([8192, 1024]), S([8192, 1024]), S([8192, 1024]), S([8192, 1024]), S([8192, 1024]), S([8192, 1024]), S([8192, 1024]), S([8192, 1024]), S([8192, 1024]), S([8192, 1024]), S([8192, 1024]), S([8192, 1024]), S([8192, 1024]), S([8192, 1024]), S([8192, 1024]), S([8192, 1024]), S([8192, 1024]), S([8192, 1024]), S([8192, 1024]), S([8192, 1024]), S([8192, 1024]), S([8192, 1024]), S([8192, 1024]))"

class Repro(torch.nn.Module):
    def forward(self, addmm_71: "f32[8192, 1024]", add_84: "f32[64, 128, 1024]", arg196_1: "f32[1024]", arg197_1: "f32[1024]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13, _shape_param_14, _shape_param_15, _shape_param_16, _shape_param_17, _shape_param_18, _shape_param_19, _shape_param_20, _shape_param_21, _shape_param_22, _shape_param_23, _shape_param_24):
        # No stacktrace found for following nodes
        view_default: "f32[64, 128, 1024]" = torch.ops.aten.view.default(addmm_71, _shape_param_0);  addmm_71 = _shape_param_0 = None
        add_tensor: "f32[64, 128, 1024]" = torch.ops.aten.add.Tensor(add_84, view_default);  add_84 = view_default = None
        var_mean_correction = torch.ops.aten.var_mean.correction(add_tensor, [2], correction = 0, keepdim = True)
        getitem: "f32[64, 128, 1]" = var_mean_correction[0]
        getitem_1: "f32[64, 128, 1]" = var_mean_correction[1];  var_mean_correction = None
        sub_tensor: "f32[64, 128, 1024]" = torch.ops.aten.sub.Tensor(add_tensor, getitem_1);  add_tensor = getitem_1 = None
        add_tensor_1: "f32[64, 128, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[64, 128, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        mul_tensor: "f32[64, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[64, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor, arg196_1);  mul_tensor = arg196_1 = None
        add_tensor_2: "f32[64, 128, 1024]" = torch.ops.aten.add.Tensor(mul_tensor_1, arg197_1);  mul_tensor_1 = arg197_1 = None
        view_default_1: "f32[8192, 1024]" = torch.ops.aten.view.default(add_tensor_2, _shape_param_1);  _shape_param_1 = None
        view_default_2: "f32[8192, 1024]" = torch.ops.aten.view.default(add_tensor_2, _shape_param_2);  _shape_param_2 = None
        view_default_3: "f32[8192, 1024]" = torch.ops.aten.view.default(add_tensor_2, _shape_param_3);  _shape_param_3 = None
        view_default_4: "f32[8192, 1024]" = torch.ops.aten.view.default(add_tensor_2, _shape_param_4);  _shape_param_4 = None
        view_default_5: "f32[8192, 1024]" = torch.ops.aten.view.default(add_tensor_2, _shape_param_5);  _shape_param_5 = None
        view_default_6: "f32[8192, 1024]" = torch.ops.aten.view.default(add_tensor_2, _shape_param_6);  _shape_param_6 = None
        view_default_7: "f32[8192, 1024]" = torch.ops.aten.view.default(add_tensor_2, _shape_param_7);  _shape_param_7 = None
        view_default_8: "f32[8192, 1024]" = torch.ops.aten.view.default(add_tensor_2, _shape_param_8);  _shape_param_8 = None
        view_default_9: "f32[8192, 1024]" = torch.ops.aten.view.default(add_tensor_2, _shape_param_9);  _shape_param_9 = None
        view_default_10: "f32[8192, 1024]" = torch.ops.aten.view.default(add_tensor_2, _shape_param_10);  _shape_param_10 = None
        view_default_11: "f32[8192, 1024]" = torch.ops.aten.view.default(add_tensor_2, _shape_param_11);  _shape_param_11 = None
        view_default_12: "f32[8192, 1024]" = torch.ops.aten.view.default(add_tensor_2, _shape_param_12);  _shape_param_12 = None
        view_default_13: "f32[8192, 1024]" = torch.ops.aten.view.default(add_tensor_2, _shape_param_13);  _shape_param_13 = None
        view_default_14: "f32[8192, 1024]" = torch.ops.aten.view.default(add_tensor_2, _shape_param_14);  _shape_param_14 = None
        view_default_15: "f32[8192, 1024]" = torch.ops.aten.view.default(add_tensor_2, _shape_param_15);  _shape_param_15 = None
        view_default_16: "f32[8192, 1024]" = torch.ops.aten.view.default(add_tensor_2, _shape_param_16);  _shape_param_16 = None
        view_default_17: "f32[8192, 1024]" = torch.ops.aten.view.default(add_tensor_2, _shape_param_17);  _shape_param_17 = None
        view_default_18: "f32[8192, 1024]" = torch.ops.aten.view.default(add_tensor_2, _shape_param_18);  _shape_param_18 = None
        view_default_19: "f32[8192, 1024]" = torch.ops.aten.view.default(add_tensor_2, _shape_param_19);  _shape_param_19 = None
        view_default_20: "f32[8192, 1024]" = torch.ops.aten.view.default(add_tensor_2, _shape_param_20);  _shape_param_20 = None
        view_default_21: "f32[8192, 1024]" = torch.ops.aten.view.default(add_tensor_2, _shape_param_21);  _shape_param_21 = None
        view_default_22: "f32[8192, 1024]" = torch.ops.aten.view.default(add_tensor_2, _shape_param_22);  _shape_param_22 = None
        view_default_23: "f32[8192, 1024]" = torch.ops.aten.view.default(add_tensor_2, _shape_param_23);  _shape_param_23 = None
        view_default_24: "f32[8192, 1024]" = torch.ops.aten.view.default(add_tensor_2, _shape_param_24);  add_tensor_2 = _shape_param_24 = None
        return (view_default_1, view_default_2, view_default_3, view_default_4, view_default_5, view_default_6, view_default_7, view_default_8, view_default_9, view_default_10, view_default_11, view_default_12, view_default_13, view_default_14, view_default_15, view_default_16, view_default_17, view_default_18, view_default_19, view_default_20, view_default_21, view_default_22, view_default_23, view_default_24)

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
