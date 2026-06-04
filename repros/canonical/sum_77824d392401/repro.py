"""
Standalone repro captured via capture_hook.
Label: genai_CrossEntropyBackward_001
Pattern hash: 77824d392401
Shape hash: 4b36b0bc
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([], bf16), T([8192], i64), T([8192, 262144], bf16), T([8192, 1], f32), T([8192, 1], f32), S([8192]), S([1, 262144]), S([8192, 262144]))"

class Repro(torch.nn.Module):
    def forward(self, arg4_1: "bf16[]", arg1_1: "i64[8192]", arg0_1: "bf16[8192, 262144]", arg2_1: "f32[8192, 1]", arg3_1: "f32[8192, 1]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        expand_default: "bf16[8192]" = torch.ops.aten.expand.default(arg4_1, _shape_param_0);  arg4_1 = _shape_param_0 = None
        unsqueeze_default: "i64[8192, 1]" = torch.ops.aten.unsqueeze.default(arg1_1, 1);  arg1_1 = None
        ne_scalar: "b8[8192, 1]" = torch.ops.aten.ne.Scalar(unsqueeze_default, -100)
        full_default: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "i64[8192, 1]" = torch.ops.aten.where.self(ne_scalar, unsqueeze_default, full_default);  unsqueeze_default = full_default = None
        iota_default: "i64[262144]" = torch.ops.prims.iota.default(262144, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        view_default: "i64[1, 262144]" = torch.ops.aten.view.default(iota_default, _shape_param_1);  iota_default = _shape_param_1 = None
        expand_default_1: "i64[8192, 262144]" = torch.ops.aten.expand.default(where_self, _shape_param_2);  where_self = _shape_param_2 = None
        eq_tensor: "b8[8192, 262144]" = torch.ops.aten.eq.Tensor(expand_default_1, view_default);  expand_default_1 = view_default = None
        scalar_tensor_default: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
        scalar_tensor_default_1: "f32[]" = torch.ops.aten.scalar_tensor.default(-1.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
        where_self_1: "f32[8192, 262144]" = torch.ops.aten.where.self(eq_tensor, scalar_tensor_default_1, scalar_tensor_default);  eq_tensor = scalar_tensor_default_1 = scalar_tensor_default = None
        unsqueeze_default_1: "bf16[8192, 1]" = torch.ops.aten.unsqueeze.default(expand_default, 1);  expand_default = None
        full_default_1: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_2: "bf16[8192, 1]" = torch.ops.aten.where.self(ne_scalar, unsqueeze_default_1, full_default_1);  ne_scalar = unsqueeze_default_1 = full_default_1 = None
        mul_tensor: "f32[8192, 262144]" = torch.ops.aten.mul.Tensor(where_self_1, where_self_2);  where_self_1 = where_self_2 = None
        convert_element_type_default: "f32[8192, 262144]" = torch.ops.prims.convert_element_type.default(mul_tensor, torch.float32);  mul_tensor = None
        convert_element_type_default_1: "f32[8192, 262144]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.float32);  arg0_1 = None
        sub_tensor: "f32[8192, 262144]" = torch.ops.aten.sub.Tensor(convert_element_type_default_1, arg2_1);  convert_element_type_default_1 = arg2_1 = None
        sub_tensor_1: "f32[8192, 262144]" = torch.ops.aten.sub.Tensor(sub_tensor, arg3_1);  sub_tensor = arg3_1 = None
        convert_element_type_default_2: "bf16[8192, 262144]" = torch.ops.prims.convert_element_type.default(sub_tensor_1, torch.bfloat16);  sub_tensor_1 = None
        convert_element_type_default_3: "f32[8192, 262144]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_2, torch.float32);  convert_element_type_default_2 = None
        exp_default: "f32[8192, 262144]" = torch.ops.aten.exp.default(convert_element_type_default_3);  convert_element_type_default_3 = None
        sum_dim_int_list: "f32[8192, 1]" = torch.ops.aten.sum.dim_IntList(convert_element_type_default, [1], True)
        mul_tensor_1: "f32[8192, 262144]" = torch.ops.aten.mul.Tensor(exp_default, sum_dim_int_list);  exp_default = sum_dim_int_list = None
        sub_tensor_2: "f32[8192, 262144]" = torch.ops.aten.sub.Tensor(convert_element_type_default, mul_tensor_1);  convert_element_type_default = mul_tensor_1 = None
        convert_element_type_default_4: "bf16[8192, 262144]" = torch.ops.prims.convert_element_type.default(sub_tensor_2, torch.bfloat16);  sub_tensor_2 = None
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
