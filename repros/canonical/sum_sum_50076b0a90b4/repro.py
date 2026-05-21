"""
Standalone repro captured via capture_hook.
Label: hf_AllenaiLongformerBase_train_005
Pattern hash: 50076b0a90b4
Shape hash: f64748a8
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
_shapes_config = "(T([8192, 768], f32), T([8192, 768], f32), T([8192, 768], f32), T([8, 1024, 768], f32), T([768], f32), T([8, 1024, 768], f32), T([8, 1024, 1], f32), T([8, 1024, 768], b8), S([1024, 8, 768]), S([1024, 8, 768]), S([1024, 8, 768]), S([8192, 768]))"

class Repro(torch.nn.Module):
    def forward(self, mm_127: "f32[8192, 768]", mm_129: "f32[8192, 768]", mm_131: "f32[8192, 768]", mul_279: "f32[8, 1024, 768]", arg8_1: "f32[768]", arg109_1: "f32[8, 1024, 768]", arg297_1: "f32[8, 1024, 1]", arg108_1: "b8[8, 1024, 768]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # No stacktrace found for following nodes
        view_default: "f32[1024, 8, 768]" = torch.ops.aten.view.default(mm_127, _shape_param_0);  mm_127 = _shape_param_0 = None
        view_default_1: "f32[1024, 8, 768]" = torch.ops.aten.view.default(mm_129, _shape_param_1);  mm_129 = _shape_param_1 = None
        add_tensor: "f32[1024, 8, 768]" = torch.ops.aten.add.Tensor(view_default, view_default_1);  view_default = view_default_1 = None
        view_default_2: "f32[1024, 8, 768]" = torch.ops.aten.view.default(mm_131, _shape_param_2);  mm_131 = _shape_param_2 = None
        add_tensor_1: "f32[1024, 8, 768]" = torch.ops.aten.add.Tensor(add_tensor, view_default_2);  add_tensor = view_default_2 = None
        permute_default: "f32[8, 1024, 768]" = torch.ops.aten.permute.default(add_tensor_1, [1, 0, 2]);  add_tensor_1 = None
        add_tensor_2: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_279, permute_default);  mul_279 = permute_default = None
        mul_tensor: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_tensor_2, arg8_1);  add_tensor_2 = arg8_1 = None
        mul_tensor_1: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, 768)
        sum_dim_int_list: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [2], True)
        mul_tensor_2: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, arg109_1);  mul_tensor = None
        sum_dim_int_list_1: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [2], True);  mul_tensor_2 = None
        mul_tensor_3: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(arg109_1, sum_dim_int_list_1);  arg109_1 = sum_dim_int_list_1 = None
        sub_tensor: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_1: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor_3);  sub_tensor = mul_tensor_3 = None
        mul_tensor_4: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(arg297_1, sub_tensor_1);  arg297_1 = sub_tensor_1 = None
        convert_element_type_default: "f32[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(arg108_1, torch.float32);  arg108_1 = None
        mul_tensor_5: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 1.0);  convert_element_type_default = None
        mul_tensor_6: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_4, mul_tensor_5);  mul_tensor_4 = mul_tensor_5 = None
        view_default_3: "f32[8192, 768]" = torch.ops.aten.view.default(mul_tensor_6, _shape_param_3);  mul_tensor_6 = _shape_param_3 = None
        return view_default_3



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
