"""
Standalone repro captured via capture_hook.
Label: inductor_timm_perf-6-6-linux.aws.a100_graph21
Pattern hash: 37e51b759e61
Shape hash: 8bb3a769
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([175360, 768], bf16), T([768], bf16), T([128, 1370, 768], bf16), T([768], bf16), T([768], bf16), S([128, 1370, 768]))"

class Repro(torch.nn.Module):
    def forward(self, addmm_47: "bf16[175360, 768]", arg172_1: "bf16[768]", add_80: "bf16[128, 1370, 768]", arg173_1: "bf16[768]", arg174_1: "bf16[768]", _shape_param_0):
        # No stacktrace found for following nodes
        view_default: "bf16[128, 1370, 768]" = torch.ops.aten.view.default(addmm_47, _shape_param_0);  addmm_47 = _shape_param_0 = None
        mul_tensor: "bf16[128, 1370, 768]" = torch.ops.aten.mul.Tensor(view_default, arg172_1);  view_default = arg172_1 = None
        add_tensor: "bf16[128, 1370, 768]" = torch.ops.aten.add.Tensor(add_80, mul_tensor);  add_80 = mul_tensor = None
        convert_element_type_default: "f32[128, 1370, 768]" = torch.ops.prims.convert_element_type.default(add_tensor, torch.float32);  add_tensor = None
        var_mean_correction = torch.ops.aten.var_mean.correction(convert_element_type_default, [2], correction = 0, keepdim = True)
        getitem: "f32[128, 1370, 1]" = var_mean_correction[0]
        getitem_1: "f32[128, 1370, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor_1: "f32[128, 1370, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-06);  getitem = None
        rsqrt_default: "f32[128, 1370, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        sub_tensor: "f32[128, 1370, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_default, getitem_1);  convert_element_type_default = getitem_1 = None
        mul_tensor_1: "f32[128, 1370, 768]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_2: "f32[128, 1370, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_1, arg173_1);  mul_tensor_1 = arg173_1 = None
        add_tensor_2: "f32[128, 1370, 768]" = torch.ops.aten.add.Tensor(mul_tensor_2, arg174_1);  mul_tensor_2 = arg174_1 = None
        convert_element_type_default_1: "bf16[128, 1370, 768]" = torch.ops.prims.convert_element_type.default(add_tensor_2, torch.bfloat16);  add_tensor_2 = None
        select_int: "bf16[128, 768]" = torch.ops.aten.select.int(convert_element_type_default_1, 1, 0);  convert_element_type_default_1 = None
        clone_default: "bf16[128, 768]" = torch.ops.aten.clone.default(select_int);  select_int = None
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
