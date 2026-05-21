"""
Standalone repro captured via capture_hook.
Label: hf_AllenaiLongformerBase_train_005
Pattern hash: 06a0926a980a
Shape hash: 9d93e2b9
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
_shapes_config = "(T([8192, 768], f32), T([8, 1024, 768], f32), T([768], f32), T([8, 1024, 768], f32), T([8, 1024, 1], f32), T([8, 1024, 768], b8), S([8, 1024, 768]), S([8192, 768]))"

class Repro(torch.nn.Module):
    def forward(self, mm_134: "f32[8192, 768]", mul_290: "f32[8, 1024, 768]", arg5_1: "f32[768]", arg104_1: "f32[8, 1024, 768]", arg298_1: "f32[8, 1024, 1]", arg103_1: "b8[8, 1024, 768]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        view_default: "f32[8, 1024, 768]" = torch.ops.aten.view.default(mm_134, _shape_param_0);  mm_134 = _shape_param_0 = None
        add_tensor: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_290, view_default);  mul_290 = view_default = None
        mul_tensor: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_tensor, arg5_1);  add_tensor = arg5_1 = None
        mul_tensor_1: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, 768)
        sum_dim_int_list: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [2], True)
        mul_tensor_2: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, arg104_1);  mul_tensor = None
        sum_dim_int_list_1: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [2], True);  mul_tensor_2 = None
        mul_tensor_3: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(arg104_1, sum_dim_int_list_1);  arg104_1 = sum_dim_int_list_1 = None
        sub_tensor: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_1: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor_3);  sub_tensor = mul_tensor_3 = None
        mul_tensor_4: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(arg298_1, sub_tensor_1);  arg298_1 = sub_tensor_1 = None
        convert_element_type_default: "f32[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(arg103_1, torch.float32);  arg103_1 = None
        mul_tensor_5: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 1.0);  convert_element_type_default = None
        mul_tensor_6: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_4, mul_tensor_5);  mul_tensor_4 = mul_tensor_5 = None
        view_default_1: "f32[8192, 768]" = torch.ops.aten.view.default(mul_tensor_6, _shape_param_1);  mul_tensor_6 = _shape_param_1 = None
        return view_default_1



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
