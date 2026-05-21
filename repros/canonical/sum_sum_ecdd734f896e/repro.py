"""
Standalone repro captured via capture_hook.
Label: hf_GoogleFnet_train_001
Pattern hash: ecdd734f896e
Shape hash: 22bbd240
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
_shapes_config = "(T([32, 512, 768, 2], f32), T([32, 512, 768], f32), T([768], f32), T([32, 512, 768], f32), T([32, 512, 1], f32), T([32, 512, 768], b8), S([16384, 768]))"

class Repro(torch.nn.Module):
    def forward(self, view_as_real_10: "f32[32, 512, 768, 2]", mul_306: "f32[32, 512, 768]", arg9_1: "f32[768]", arg65_1: "f32[32, 512, 768]", arg163_1: "f32[32, 512, 1]", arg64_1: "b8[32, 512, 768]", _shape_param_0):
        # No stacktrace found for following nodes
        select_int: "f32[32, 512, 768]" = torch.ops.aten.select.int(view_as_real_10, 3, 0);  view_as_real_10 = None
        add_tensor: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_306, select_int);  mul_306 = select_int = None
        mul_tensor: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_tensor, arg9_1);  add_tensor = arg9_1 = None
        mul_tensor_1: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, 768)
        sum_dim_int_list: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [2], True)
        mul_tensor_2: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, arg65_1);  mul_tensor = None
        sum_dim_int_list_1: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [2], True);  mul_tensor_2 = None
        mul_tensor_3: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(arg65_1, sum_dim_int_list_1);  arg65_1 = sum_dim_int_list_1 = None
        sub_tensor: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_1: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor_3);  sub_tensor = mul_tensor_3 = None
        mul_tensor_4: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(arg163_1, sub_tensor_1);  arg163_1 = sub_tensor_1 = None
        convert_element_type_default: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(arg64_1, torch.float32);  arg64_1 = None
        mul_tensor_5: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 1.0);  convert_element_type_default = None
        mul_tensor_6: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_4, mul_tensor_5);  mul_tensor_4 = mul_tensor_5 = None
        view_default: "f32[16384, 768]" = torch.ops.aten.view.default(mul_tensor_6, _shape_param_0);  mul_tensor_6 = _shape_param_0 = None
        return view_default



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
