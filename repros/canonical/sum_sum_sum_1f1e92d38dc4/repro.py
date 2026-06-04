"""
Standalone repro captured via capture_hook.
Label: hf_DebertaV2ForMaskedLM_train_001
Pattern hash: 1f1e92d38dc4
Shape hash: 880b1002
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([4096, 1536], f32), T([8, 512, 1536], f32), T([4096, 1536], f32), T([4096, 1536], f32), T([8, 512, 1536], b8), T([1536], f32), T([8, 512, 1536], f32), T([1, 512, 1536], f32), T([8, 512, 1], f32), T([8, 512, 1], f32), T([1, 512], i64, gen=Index(512)), T([], f32), T([8, 512], i64, gen=Index(128100)), T([128100, 1536], f32), S([8, 512, 1536]), S([8, 512, 1536]), S([8, 512, 1536]))"

class Repro(torch.nn.Module):
    def forward(self, mm_286: "f32[4096, 1536]", mul_635: "f32[8, 512, 1536]", mm_288: "f32[4096, 1536]", mm_290: "f32[4096, 1536]", arg203_1: "b8[8, 512, 1536]", arg3_1: "f32[1536]", arg199_1: "f32[8, 512, 1536]", arg200_1: "f32[1, 512, 1536]", arg201_1: "f32[8, 512, 1]", arg202_1: "f32[8, 512, 1]", arg1_1: "i64[1, 512]", full_1: "f32[]", arg0_1: "i64[8, 512]", mm_1: "f32[128100, 1536]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        view_default: "f32[8, 512, 1536]" = torch.ops.aten.view.default(mm_286, _shape_param_0);  mm_286 = _shape_param_0 = None
        add_tensor: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_635, view_default);  mul_635 = view_default = None
        view_default_1: "f32[8, 512, 1536]" = torch.ops.aten.view.default(mm_288, _shape_param_1);  mm_288 = _shape_param_1 = None
        add_tensor_1: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(add_tensor, view_default_1);  add_tensor = view_default_1 = None
        view_default_2: "f32[8, 512, 1536]" = torch.ops.aten.view.default(mm_290, _shape_param_2);  mm_290 = _shape_param_2 = None
        add_tensor_2: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(add_tensor_1, view_default_2);  add_tensor_1 = view_default_2 = None
        convert_element_type_default: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(arg203_1, torch.float32);  arg203_1 = None
        mul_tensor: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 1.1111111111111112);  convert_element_type_default = None
        mul_tensor_1: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_tensor_2, mul_tensor);  add_tensor_2 = mul_tensor = None
        mul_tensor_2: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_tensor_1, arg3_1);  arg3_1 = None
        mul_tensor_3: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_tensor_2, 1536)
        sum_dim_int_list: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [2], True)
        add_tensor_3: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(arg199_1, arg200_1);  arg199_1 = arg200_1 = None
        sub_tensor: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(add_tensor_3, arg201_1);  add_tensor_3 = arg201_1 = None
        mul_tensor_4: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_tensor, arg202_1);  sub_tensor = None
        mul_tensor_5: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_tensor_2, mul_tensor_4);  mul_tensor_2 = None
        sum_dim_int_list_1: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_5, [2], True);  mul_tensor_5 = None
        mul_tensor_6: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_tensor_4, sum_dim_int_list_1);  sum_dim_int_list_1 = None
        sub_tensor_1: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(mul_tensor_3, sum_dim_int_list);  mul_tensor_3 = sum_dim_int_list = None
        sub_tensor_2: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(sub_tensor_1, mul_tensor_6);  sub_tensor_1 = mul_tensor_6 = None
        div_tensor: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(arg202_1, 1536);  arg202_1 = None
        mul_tensor_7: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(div_tensor, sub_tensor_2);  div_tensor = sub_tensor_2 = None
        mul_tensor_8: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_tensor_1, mul_tensor_4);  mul_tensor_4 = None
        sum_dim_int_list_2: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_tensor_8, [0, 1]);  mul_tensor_8 = None
        sum_dim_int_list_3: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_tensor_1, [0, 1]);  mul_tensor_1 = None
        sum_dim_int_list_4: "f32[1, 512, 1536]" = torch.ops.aten.sum.dim_IntList(mul_tensor_7, [0], True)
        eq_scalar: "b8[1, 512]" = torch.ops.aten.eq.Scalar(arg1_1, -1)
        unsqueeze_default: "b8[1, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_scalar, -1);  eq_scalar = None
        where_self: "f32[1, 512, 1536]" = torch.ops.aten.where.self(unsqueeze_default, full_1, sum_dim_int_list_4);  unsqueeze_default = sum_dim_int_list_4 = None
        full_default: "f32[512, 1536]" = torch.ops.aten.full.default([512, 1536], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default: "f32[512, 1536]" = torch.ops.aten.index_put.default(full_default, [arg1_1], where_self, True);  full_default = arg1_1 = where_self = None
        eq_scalar_1: "b8[8, 512]" = torch.ops.aten.eq.Scalar(arg0_1, 0)
        unsqueeze_default_1: "b8[8, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_scalar_1, -1);  eq_scalar_1 = None
        where_self_1: "f32[8, 512, 1536]" = torch.ops.aten.where.self(unsqueeze_default_1, full_1, mul_tensor_7);  unsqueeze_default_1 = full_1 = mul_tensor_7 = None
        full_default_1: "f32[128100, 1536]" = torch.ops.aten.full.default([128100, 1536], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default_1: "f32[128100, 1536]" = torch.ops.aten.index_put.default(full_default_1, [arg0_1], where_self_1, True);  full_default_1 = arg0_1 = where_self_1 = None
        add_tensor_4: "f32[128100, 1536]" = torch.ops.aten.add.Tensor(mm_1, index_put_default_1);  mm_1 = index_put_default_1 = None
        return (sum_dim_int_list_2, sum_dim_int_list_3, index_put_default, add_tensor_4)

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
