"""
Standalone repro captured via capture_hook.
Label: hf_GPTJForCausalLM_train_001
Pattern hash: 9e60972dd685
Shape hash: 915ac49d
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([128, 4096], f32), T([128, 4096], f32), T([128, 4096], f32), T([128, 4096], f32), T([4096], f32), T([1, 128, 4096], f32), T([1, 128, 1], f32), T([1, 128, 1], f32), T([1, 128, 4096], f32), T([1, 128], i64, gen=Index(50400)), T([], f32), S([1, 128, 4096]), S([1, 128, 4096]), S([1, 128, 4096]), S([1, 128, 4096]))"

class Repro(torch.nn.Module):
    def forward(self, mm_328: "f32[128, 4096]", mm_333: "f32[128, 4096]", mm_335: "f32[128, 4096]", mm_337: "f32[128, 4096]", arg1_1: "f32[4096]", arg199_1: "f32[1, 128, 4096]", arg201_1: "f32[1, 128, 1]", arg202_1: "f32[1, 128, 1]", add_378: "f32[1, 128, 4096]", arg0_1: "i64[1, 128]", full_1: "f32[]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # No stacktrace found for following nodes
        view_default: "f32[1, 128, 4096]" = torch.ops.aten.view.default(mm_328, _shape_param_0);  mm_328 = _shape_param_0 = None
        view_default_1: "f32[1, 128, 4096]" = torch.ops.aten.view.default(mm_333, _shape_param_1);  mm_333 = _shape_param_1 = None
        add_tensor: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_default, view_default_1);  view_default = view_default_1 = None
        view_default_2: "f32[1, 128, 4096]" = torch.ops.aten.view.default(mm_335, _shape_param_2);  mm_335 = _shape_param_2 = None
        add_tensor_1: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_tensor, view_default_2);  add_tensor = view_default_2 = None
        view_default_3: "f32[1, 128, 4096]" = torch.ops.aten.view.default(mm_337, _shape_param_3);  mm_337 = _shape_param_3 = None
        add_tensor_2: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_tensor_1, view_default_3);  add_tensor_1 = view_default_3 = None
        mul_tensor: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_tensor_2, arg1_1);  arg1_1 = None
        mul_tensor_1: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_tensor, 4096)
        sum_dim_int_list: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [2], True)
        sub_tensor: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(arg199_1, arg201_1);  arg199_1 = arg201_1 = None
        mul_tensor_2: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub_tensor, arg202_1);  sub_tensor = None
        mul_tensor_3: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_tensor, mul_tensor_2);  mul_tensor = None
        sum_dim_int_list_1: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_3, [2], True);  mul_tensor_3 = None
        mul_tensor_4: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_tensor_2, sum_dim_int_list_1);  sum_dim_int_list_1 = None
        sub_tensor_1: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_2: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(sub_tensor_1, mul_tensor_4);  sub_tensor_1 = mul_tensor_4 = None
        div_tensor: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(arg202_1, 4096);  arg202_1 = None
        mul_tensor_5: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(div_tensor, sub_tensor_2);  div_tensor = sub_tensor_2 = None
        mul_tensor_6: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_tensor_2, mul_tensor_2);  mul_tensor_2 = None
        sum_dim_int_list_2: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor_6, [0, 1]);  mul_tensor_6 = None
        sum_dim_int_list_3: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_tensor_2, [0, 1]);  add_tensor_2 = None
        add_tensor_3: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_378, mul_tensor_5);  add_378 = mul_tensor_5 = None
        eq_scalar: "b8[1, 128]" = torch.ops.aten.eq.Scalar(arg0_1, -1)
        unsqueeze_default: "b8[1, 128, 1]" = torch.ops.aten.unsqueeze.default(eq_scalar, -1);  eq_scalar = None
        where_self: "f32[1, 128, 4096]" = torch.ops.aten.where.self(unsqueeze_default, full_1, add_tensor_3);  unsqueeze_default = full_1 = add_tensor_3 = None
        full_default: "f32[50400, 4096]" = torch.ops.aten.full.default([50400, 4096], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default: "f32[50400, 4096]" = torch.ops.aten.index_put.default(full_default, [arg0_1], where_self, True);  full_default = arg0_1 = where_self = None
        return (sum_dim_int_list_2, sum_dim_int_list_3, index_put_default)

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
