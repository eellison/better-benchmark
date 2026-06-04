"""
Standalone repro captured via capture_hook.
Label: hf_GPTNeoForCausalLM_train_001
Pattern hash: fefa336f7504
Shape hash: a065e161
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([50260, 2048], f32), T([4096, 2048], f32), T([4096, 2048], f32), T([4096, 2048], f32), T([2048], f32), T([32, 128, 2048], f32), T([1, 128, 2048], f32), T([32, 128, 1], f32), T([32, 128, 1], f32), T([32, 128, 2048], f32), T([1, 128], i64, gen=Index(2048)), T([], f32), T([32, 128], i64, gen=Index(50257)), S([32, 128, 2048]), S([32, 128, 2048]), S([32, 128, 2048]))"

class Repro(torch.nn.Module):
    def forward(self, mm: "f32[50260, 2048]", mm_285: "f32[4096, 2048]", mm_287: "f32[4096, 2048]", mm_289: "f32[4096, 2048]", arg2_1: "f32[2048]", arg219_1: "f32[32, 128, 2048]", arg221_1: "f32[1, 128, 2048]", arg222_1: "f32[32, 128, 1]", arg223_1: "f32[32, 128, 1]", add_189: "f32[32, 128, 2048]", arg220_1: "i64[1, 128]", full_1: "f32[]", arg0_1: "i64[32, 128]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        slice_tensor: "f32[50257, 2048]" = torch.ops.aten.slice.Tensor(mm, 0, 0, -3);  mm = None
        view_default: "f32[32, 128, 2048]" = torch.ops.aten.view.default(mm_285, _shape_param_0);  mm_285 = _shape_param_0 = None
        view_default_1: "f32[32, 128, 2048]" = torch.ops.aten.view.default(mm_287, _shape_param_1);  mm_287 = _shape_param_1 = None
        add_tensor: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(view_default, view_default_1);  view_default = view_default_1 = None
        view_default_2: "f32[32, 128, 2048]" = torch.ops.aten.view.default(mm_289, _shape_param_2);  mm_289 = _shape_param_2 = None
        add_tensor_1: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_tensor, view_default_2);  add_tensor = view_default_2 = None
        mul_tensor: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_tensor_1, arg2_1);  arg2_1 = None
        mul_tensor_1: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_tensor, 2048)
        sum_dim_int_list: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [2], True)
        add_tensor_2: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(arg219_1, arg221_1);  arg219_1 = arg221_1 = None
        sub_tensor: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(add_tensor_2, arg222_1);  add_tensor_2 = arg222_1 = None
        mul_tensor_2: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_tensor, arg223_1);  sub_tensor = None
        mul_tensor_3: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_tensor, mul_tensor_2);  mul_tensor = None
        sum_dim_int_list_1: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_3, [2], True);  mul_tensor_3 = None
        mul_tensor_4: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_tensor_2, sum_dim_int_list_1);  sum_dim_int_list_1 = None
        sub_tensor_1: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_2: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_tensor_1, mul_tensor_4);  sub_tensor_1 = mul_tensor_4 = None
        div_tensor: "f32[32, 128, 1]" = torch.ops.aten.div.Tensor(arg223_1, 2048);  arg223_1 = None
        mul_tensor_5: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(div_tensor, sub_tensor_2);  div_tensor = sub_tensor_2 = None
        mul_tensor_6: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_tensor_1, mul_tensor_2);  mul_tensor_2 = None
        sum_dim_int_list_2: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_tensor_6, [0, 1]);  mul_tensor_6 = None
        sum_dim_int_list_3: "f32[2048]" = torch.ops.aten.sum.dim_IntList(add_tensor_1, [0, 1]);  add_tensor_1 = None
        add_tensor_3: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_189, mul_tensor_5);  add_189 = mul_tensor_5 = None
        sum_dim_int_list_4: "f32[1, 128, 2048]" = torch.ops.aten.sum.dim_IntList(add_tensor_3, [0], True)
        eq_scalar: "b8[1, 128]" = torch.ops.aten.eq.Scalar(arg220_1, -1)
        unsqueeze_default: "b8[1, 128, 1]" = torch.ops.aten.unsqueeze.default(eq_scalar, -1);  eq_scalar = None
        where_self: "f32[1, 128, 2048]" = torch.ops.aten.where.self(unsqueeze_default, full_1, sum_dim_int_list_4);  unsqueeze_default = sum_dim_int_list_4 = None
        full_default: "f32[2048, 2048]" = torch.ops.aten.full.default([2048, 2048], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default: "f32[2048, 2048]" = torch.ops.aten.index_put.default(full_default, [arg220_1], where_self, True);  full_default = arg220_1 = where_self = None
        eq_scalar_1: "b8[32, 128]" = torch.ops.aten.eq.Scalar(arg0_1, -1)
        unsqueeze_default_1: "b8[32, 128, 1]" = torch.ops.aten.unsqueeze.default(eq_scalar_1, -1);  eq_scalar_1 = None
        where_self_1: "f32[32, 128, 2048]" = torch.ops.aten.where.self(unsqueeze_default_1, full_1, add_tensor_3);  unsqueeze_default_1 = full_1 = add_tensor_3 = None
        full_default_1: "f32[50257, 2048]" = torch.ops.aten.full.default([50257, 2048], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default_1: "f32[50257, 2048]" = torch.ops.aten.index_put.default(full_default_1, [arg0_1], where_self_1, True);  full_default_1 = arg0_1 = where_self_1 = None
        add_tensor_4: "f32[50257, 2048]" = torch.ops.aten.add.Tensor(slice_tensor, index_put_default_1);  slice_tensor = index_put_default_1 = None
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
