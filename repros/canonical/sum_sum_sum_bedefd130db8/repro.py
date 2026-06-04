"""
Standalone repro captured via capture_hook.
Label: hf_GPTNeoForCausalLM_train_001
Pattern hash: bedefd130db8
Shape hash: 460a5ebb
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([4096, 2048], f32), T([2048], f32), T([32, 128, 2048], f32), T([32, 128, 1], f32), T([32, 128, 2048], f32), S([32, 128, 2048]), S([4096, 2048]), S([2048]))"

class Repro(torch.nn.Module):
    def forward(self, mm_280: "f32[4096, 2048]", arg8_1: "f32[2048]", arg227_1: "f32[32, 128, 2048]", arg538_1: "f32[32, 128, 1]", add_184: "f32[32, 128, 2048]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        view_default: "f32[32, 128, 2048]" = torch.ops.aten.view.default(mm_280, _shape_param_0);  mm_280 = _shape_param_0 = None
        mul_tensor: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(view_default, arg8_1);  arg8_1 = None
        mul_tensor_1: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_tensor, 2048)
        sum_dim_int_list: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [2], True)
        mul_tensor_2: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_tensor, arg227_1);  mul_tensor = None
        sum_dim_int_list_1: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [2], True);  mul_tensor_2 = None
        mul_tensor_3: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg227_1, sum_dim_int_list_1);  sum_dim_int_list_1 = None
        sub_tensor: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_1: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor_3);  sub_tensor = mul_tensor_3 = None
        mul_tensor_4: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg538_1, sub_tensor_1);  arg538_1 = sub_tensor_1 = None
        mul_tensor_5: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(view_default, arg227_1);  arg227_1 = None
        sum_dim_int_list_2: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_tensor_5, [0, 1]);  mul_tensor_5 = None
        sum_dim_int_list_3: "f32[2048]" = torch.ops.aten.sum.dim_IntList(view_default, [0, 1]);  view_default = None
        add_tensor: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_184, mul_tensor_4);  add_184 = mul_tensor_4 = None
        view_default_1: "f32[4096, 2048]" = torch.ops.aten.view.default(add_tensor, _shape_param_1);  add_tensor = _shape_param_1 = None
        permute_default: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_default_1, [1, 0])
        sum_dim_int_list_4: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_default_1, [0], True);  view_default_1 = None
        view_default_2: "f32[2048]" = torch.ops.aten.view.default(sum_dim_int_list_4, _shape_param_2);  sum_dim_int_list_4 = _shape_param_2 = None
        return (sum_dim_int_list_2, sum_dim_int_list_3, permute_default, view_default_2)

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
