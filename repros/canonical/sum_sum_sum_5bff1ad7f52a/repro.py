"""
Standalone repro captured via capture_hook.
Label: genai_LayerNormBackward_001
Pattern hash: 5bff1ad7f52a
Shape hash: e45e1d05
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
_shapes_config = "(T([], bf16), T([512], f32), T([1152000, 512], bf16), T([1152000, 1], f32), T([1152000, 1], f32), S([1152000, 512]))"

class Repro(torch.nn.Module):
    def forward(self, arg4_1: "bf16[]", arg1_1: "f32[512]", arg0_1: "bf16[1152000, 512]", arg2_1: "f32[1152000, 1]", arg3_1: "f32[1152000, 1]", _shape_param_0):
        # No stacktrace found for following nodes
        expand_default: "bf16[1152000, 512]" = torch.ops.aten.expand.default(arg4_1, _shape_param_0);  arg4_1 = _shape_param_0 = None
        convert_element_type_default: "f32[1152000, 512]" = torch.ops.prims.convert_element_type.default(expand_default, torch.float32);  expand_default = None
        mul_tensor: "f32[1152000, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_default, arg1_1);  arg1_1 = None
        mul_tensor_1: "f32[1152000, 512]" = torch.ops.aten.mul.Tensor(mul_tensor, 512)
        sum_dim_int_list: "f32[1152000, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [1], True)
        convert_element_type_default_1: "f32[1152000, 512]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.float32);  arg0_1 = None
        sub_tensor: "f32[1152000, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_default_1, arg2_1);  convert_element_type_default_1 = arg2_1 = None
        mul_tensor_2: "f32[1152000, 512]" = torch.ops.aten.mul.Tensor(sub_tensor, arg3_1);  sub_tensor = None
        mul_tensor_3: "f32[1152000, 512]" = torch.ops.aten.mul.Tensor(mul_tensor, mul_tensor_2);  mul_tensor = None
        sum_dim_int_list_1: "f32[1152000, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_3, [1], True);  mul_tensor_3 = None
        mul_tensor_4: "f32[1152000, 512]" = torch.ops.aten.mul.Tensor(mul_tensor_2, sum_dim_int_list_1);  sum_dim_int_list_1 = None
        sub_tensor_1: "f32[1152000, 512]" = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_2: "f32[1152000, 512]" = torch.ops.aten.sub.Tensor(sub_tensor_1, mul_tensor_4);  sub_tensor_1 = mul_tensor_4 = None
        div_tensor: "f32[1152000, 1]" = torch.ops.aten.div.Tensor(arg3_1, 512);  arg3_1 = None
        mul_tensor_5: "f32[1152000, 512]" = torch.ops.aten.mul.Tensor(div_tensor, sub_tensor_2);  div_tensor = sub_tensor_2 = None
        mul_tensor_6: "f32[1152000, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_default, mul_tensor_2);  convert_element_type_default = mul_tensor_2 = None
        sum_dim_int_list_2: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_6, [0]);  mul_tensor_6 = None
        convert_element_type_default_2: "bf16[1152000, 512]" = torch.ops.prims.convert_element_type.default(mul_tensor_5, torch.bfloat16);  mul_tensor_5 = None
        return (sum_dim_int_list_2, convert_element_type_default_2)



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
