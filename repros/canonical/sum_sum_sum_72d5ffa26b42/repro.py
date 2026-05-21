"""
Standalone repro captured via capture_hook.
Label: torchbench_hf_Longformer_train_006
Pattern hash: 72d5ffa26b42
Shape hash: e8e0f483
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
_shapes_config = "(T([2, 1024, 768], b8), T([2, 1024, 768], f32), T([768], f32), T([2, 1024, 768], f32), T([2, 1024, 1], f32), T([2, 1024], i64, gen=Index(1)), T([2, 1024], i64, gen=Index(4098)), T([2, 1024], i64, gen=Index(50265)))"

class Repro(torch.nn.Module):
    def forward(self, arg5_1: "b8[2, 1024, 768]", arg7_1: "f32[2, 1024, 768]", arg1_1: "f32[768]", arg4_1: "f32[2, 1024, 768]", arg6_1: "f32[2, 1024, 1]", arg2_1: "i64[2, 1024]", arg3_1: "i64[2, 1024]", arg0_1: "i64[2, 1024]"):
        # No stacktrace found for following nodes
        convert_element_type_default: "f32[2, 1024, 768]" = torch.ops.prims.convert_element_type.default(arg5_1, torch.float32);  arg5_1 = None
        mul_tensor: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 1.1111111111111112);  convert_element_type_default = None
        mul_tensor_1: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(arg7_1, mul_tensor);  arg7_1 = mul_tensor = None
        mul_tensor_2: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_1, arg1_1);  arg1_1 = None
        mul_tensor_3: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_2, 768)
        sum_dim_int_list: "f32[2, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [2], True)
        mul_tensor_4: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_2, arg4_1);  mul_tensor_2 = None
        sum_dim_int_list_1: "f32[2, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_4, [2], True);  mul_tensor_4 = None
        mul_tensor_5: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(arg4_1, sum_dim_int_list_1);  sum_dim_int_list_1 = None
        sub_tensor: "f32[2, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_tensor_3, sum_dim_int_list);  mul_tensor_3 = sum_dim_int_list = None
        sub_tensor_1: "f32[2, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor_5);  sub_tensor = mul_tensor_5 = None
        mul_tensor_6: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(arg6_1, sub_tensor_1);  arg6_1 = sub_tensor_1 = None
        mul_tensor_7: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_1, arg4_1);  arg4_1 = None
        sum_dim_int_list_2: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_7, [0, 1]);  mul_tensor_7 = None
        sum_dim_int_list_3: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_1, [0, 1]);  mul_tensor_1 = None
        full_default: "b8[2, 1024, 1]" = torch.ops.aten.full.default([2, 1024, 1], False, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "f32[2, 1024, 768]" = torch.ops.aten.where.self(full_default, full_default_1, mul_tensor_6);  full_default = None
        full_default_2: "f32[1, 768]" = torch.ops.aten.full.default([1, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default: "f32[1, 768]" = torch.ops.aten.index_put.default(full_default_2, [arg2_1], where_self, True);  full_default_2 = arg2_1 = where_self = None
        eq_scalar: "b8[2, 1024]" = torch.ops.aten.eq.Scalar(arg3_1, 1)
        unsqueeze_default: "b8[2, 1024, 1]" = torch.ops.aten.unsqueeze.default(eq_scalar, -1);  eq_scalar = None
        where_self_1: "f32[2, 1024, 768]" = torch.ops.aten.where.self(unsqueeze_default, full_default_1, mul_tensor_6);  unsqueeze_default = None
        full_default_3: "f32[4098, 768]" = torch.ops.aten.full.default([4098, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default_1: "f32[4098, 768]" = torch.ops.aten.index_put.default(full_default_3, [arg3_1], where_self_1, True);  full_default_3 = arg3_1 = where_self_1 = None
        eq_scalar_1: "b8[2, 1024]" = torch.ops.aten.eq.Scalar(arg0_1, 1)
        unsqueeze_default_1: "b8[2, 1024, 1]" = torch.ops.aten.unsqueeze.default(eq_scalar_1, -1);  eq_scalar_1 = None
        where_self_2: "f32[2, 1024, 768]" = torch.ops.aten.where.self(unsqueeze_default_1, full_default_1, mul_tensor_6);  unsqueeze_default_1 = full_default_1 = mul_tensor_6 = None
        full_default_4: "f32[50265, 768]" = torch.ops.aten.full.default([50265, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default_2: "f32[50265, 768]" = torch.ops.aten.index_put.default(full_default_4, [arg0_1], where_self_2, True);  full_default_4 = arg0_1 = where_self_2 = None
        return (sum_dim_int_list_2, sum_dim_int_list_3, index_put_default, index_put_default_1, index_put_default_2)



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
