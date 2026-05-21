"""
Standalone repro captured via capture_hook.
Label: genai_RMSNormBackward_001
Pattern hash: 0324c264adf7
Shape hash: 3568a7f6
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
_shapes_config = "(T([], bf16), T([1152000, 512], bf16), T([1152000, 1], f32), T([512], f32), S([1152000, 512]), S([512]), S([1152000, 512]))"

class Repro(torch.nn.Module):
    def forward(self, arg3_1: "bf16[]", arg0_1: "bf16[1152000, 512]", arg2_1: "f32[1152000, 1]", arg1_1: "f32[512]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        expand_default: "bf16[1152000, 512]" = torch.ops.aten.expand.default(arg3_1, _shape_param_0);  arg3_1 = _shape_param_0 = None
        convert_element_type_default: "f32[1152000, 512]" = torch.ops.prims.convert_element_type.default(expand_default, torch.float32);  expand_default = None
        convert_element_type_default_1: "f32[1152000, 512]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.float32);  arg0_1 = None
        mul_tensor: "f32[1152000, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_default_1, arg2_1)
        mul_tensor_1: "f32[1152000, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_default, mul_tensor);  mul_tensor = None
        mul_tensor_2: "f32[1152000, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_default, arg1_1);  convert_element_type_default = arg1_1 = None
        sum_dim_int_list: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_1, [0], True);  mul_tensor_1 = None
        view_default: "f32[512]" = torch.ops.aten.view.default(sum_dim_int_list, _shape_param_1);  sum_dim_int_list = _shape_param_1 = None
        mul_tensor_3: "f32[1152000, 512]" = torch.ops.aten.mul.Tensor(mul_tensor_2, convert_element_type_default_1)
        mul_tensor_4: "f32[1152000, 512]" = torch.ops.aten.mul.Tensor(mul_tensor_2, arg2_1);  mul_tensor_2 = None
        sum_dim_int_list_1: "f32[1152000, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_3, [1], True);  mul_tensor_3 = None
        pow_tensor_scalar: "f32[1152000, 1]" = torch.ops.aten.pow.Tensor_Scalar(arg2_1, 3);  arg2_1 = None
        mul_scalar: "f32[1152000, 1]" = torch.ops.aten.mul.Scalar(sum_dim_int_list_1, -0.5);  sum_dim_int_list_1 = None
        mul_tensor_5: "f32[1152000, 1]" = torch.ops.aten.mul.Tensor(mul_scalar, pow_tensor_scalar);  mul_scalar = pow_tensor_scalar = None
        expand_default_1: "f32[1152000, 512]" = torch.ops.aten.expand.default(mul_tensor_5, _shape_param_2);  mul_tensor_5 = _shape_param_2 = None
        div_scalar: "f32[1152000, 512]" = torch.ops.aten.div.Scalar(expand_default_1, 512);  expand_default_1 = None
        pow_tensor_scalar_1: "f32[1152000, 512]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_default_1, 1.0);  convert_element_type_default_1 = None
        mul_scalar_1: "f32[1152000, 512]" = torch.ops.aten.mul.Scalar(pow_tensor_scalar_1, 2.0);  pow_tensor_scalar_1 = None
        mul_tensor_6: "f32[1152000, 512]" = torch.ops.aten.mul.Tensor(div_scalar, mul_scalar_1);  div_scalar = mul_scalar_1 = None
        add_tensor: "f32[1152000, 512]" = torch.ops.aten.add.Tensor(mul_tensor_4, mul_tensor_6);  mul_tensor_4 = mul_tensor_6 = None
        convert_element_type_default_2: "bf16[1152000, 512]" = torch.ops.prims.convert_element_type.default(add_tensor, torch.bfloat16);  add_tensor = None
        return (view_default, convert_element_type_default_2)



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
