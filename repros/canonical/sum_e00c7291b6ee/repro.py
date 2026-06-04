"""
Standalone repro captured via capture_hook.
Label: genai_SoftmaxBackward_001
Pattern hash: e00c7291b6ee
Shape hash: 680a04b0
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([], bf16), T([8192, 262144], bf16), T([8192, 1], f32), T([8192, 1], f32), S([8192, 262144]))"

class Repro(torch.nn.Module):
    def forward(self, arg3_1: "bf16[]", arg0_1: "bf16[8192, 262144]", arg1_1: "f32[8192, 1]", arg2_1: "f32[8192, 1]", _shape_param_0):
        # No stacktrace found for following nodes
        expand_default: "bf16[8192, 262144]" = torch.ops.aten.expand.default(arg3_1, _shape_param_0);  arg3_1 = _shape_param_0 = None
        convert_element_type_default: "f32[8192, 262144]" = torch.ops.prims.convert_element_type.default(expand_default, torch.float32);  expand_default = None
        convert_element_type_default_1: "f32[8192, 262144]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.float32);  arg0_1 = None
        sub_tensor: "f32[8192, 262144]" = torch.ops.aten.sub.Tensor(convert_element_type_default_1, arg1_1);  convert_element_type_default_1 = arg1_1 = None
        exp_default: "f32[8192, 262144]" = torch.ops.aten.exp.default(sub_tensor);  sub_tensor = None
        div_tensor: "f32[8192, 262144]" = torch.ops.aten.div.Tensor(exp_default, arg2_1);  exp_default = arg2_1 = None
        convert_element_type_default_2: "bf16[8192, 262144]" = torch.ops.prims.convert_element_type.default(div_tensor, torch.bfloat16);  div_tensor = None
        convert_element_type_default_3: "f32[8192, 262144]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_2, torch.float32);  convert_element_type_default_2 = None
        mul_tensor: "f32[8192, 262144]" = torch.ops.aten.mul.Tensor(convert_element_type_default, convert_element_type_default_3);  convert_element_type_default = None
        sum_dim_int_list: "f32[8192, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [-1], True)
        neg_default: "f32[8192, 262144]" = torch.ops.aten.neg.default(convert_element_type_default_3);  convert_element_type_default_3 = None
        fma_default: "f32[8192, 262144]" = torch.ops.prims.fma.default(neg_default, sum_dim_int_list, mul_tensor);  neg_default = sum_dim_int_list = mul_tensor = None
        convert_element_type_default_4: "bf16[8192, 262144]" = torch.ops.prims.convert_element_type.default(fma_default, torch.bfloat16);  fma_default = None
        return convert_element_type_default_4

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
