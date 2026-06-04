"""
Standalone repro captured via capture_hook.
Label: torchbench_timm_vision_transformer_infer_000
Pattern hash: 1c7b809d512d
Shape hash: 8c137207
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([6304, 384], f16), T([32, 197, 384], f16), T([384], f16), T([384], f16), S([32, 197, 384]))"

class Repro(torch.nn.Module):
    def forward(self, addmm_47: "f16[6304, 384]", add_80: "f16[32, 197, 384]", arg149_1: "f16[384]", arg150_1: "f16[384]", _shape_param_0):
        # No stacktrace found for following nodes
        view_default: "f16[32, 197, 384]" = torch.ops.aten.view.default(addmm_47, _shape_param_0);  addmm_47 = _shape_param_0 = None
        add_tensor: "f16[32, 197, 384]" = torch.ops.aten.add.Tensor(add_80, view_default);  add_80 = view_default = None
        convert_element_type_default: "f32[32, 197, 384]" = torch.ops.prims.convert_element_type.default(add_tensor, torch.float32);  add_tensor = None
        var_mean_correction = torch.ops.aten.var_mean.correction(convert_element_type_default, [2], correction = 0, keepdim = True)
        getitem: "f32[32, 197, 1]" = var_mean_correction[0]
        getitem_1: "f32[32, 197, 1]" = var_mean_correction[1];  var_mean_correction = None
        sub_tensor: "f32[32, 197, 384]" = torch.ops.aten.sub.Tensor(convert_element_type_default, getitem_1);  convert_element_type_default = getitem_1 = None
        add_tensor_1: "f32[32, 197, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-06);  getitem = None
        rsqrt_default: "f32[32, 197, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        mul_tensor: "f32[32, 197, 384]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[32, 197, 384]" = torch.ops.aten.mul.Tensor(mul_tensor, arg149_1);  mul_tensor = arg149_1 = None
        add_tensor_2: "f32[32, 197, 384]" = torch.ops.aten.add.Tensor(mul_tensor_1, arg150_1);  mul_tensor_1 = arg150_1 = None
        convert_element_type_default_1: "f16[32, 197, 384]" = torch.ops.prims.convert_element_type.default(add_tensor_2, torch.float16);  add_tensor_2 = None
        select_int: "f16[32, 384]" = torch.ops.aten.select.int(convert_element_type_default_1, 1, 0);  convert_element_type_default_1 = None
        clone_default: "f16[32, 384]" = torch.ops.aten.clone.default(select_int);  select_int = None
        return clone_default

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
