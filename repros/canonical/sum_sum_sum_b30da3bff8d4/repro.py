"""
Standalone repro captured via capture_hook.
Label: hf_GoogleFnet_train_001
Pattern hash: b30da3bff8d4
Shape hash: a1091d37
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
_shapes_config = "(T([16384, 768], f32), T([32, 512, 768], f32), T([768], f32), T([32, 512, 768], f32), T([32, 512, 1], f32), T([32, 512, 768, 2], f32), S([32, 512, 768]))"

class Repro(torch.nn.Module):
    def forward(self, mm_50: "f32[16384, 768]", mul_312: "f32[32, 512, 768]", arg6_1: "f32[768]", arg60_1: "f32[32, 512, 768]", arg164_1: "f32[32, 512, 1]", full_2: "f32[32, 512, 768, 2]", _shape_param_0):
        # No stacktrace found for following nodes
        view_default: "f32[32, 512, 768]" = torch.ops.aten.view.default(mm_50, _shape_param_0);  mm_50 = _shape_param_0 = None
        add_tensor: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_312, view_default);  mul_312 = view_default = None
        mul_tensor: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_tensor, arg6_1);  arg6_1 = None
        mul_tensor_1: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, 768)
        sum_dim_int_list: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [2], True)
        mul_tensor_2: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, arg60_1);  mul_tensor = None
        sum_dim_int_list_1: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [2], True);  mul_tensor_2 = None
        mul_tensor_3: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(arg60_1, sum_dim_int_list_1);  sum_dim_int_list_1 = None
        sub_tensor: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_1: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor_3);  sub_tensor = mul_tensor_3 = None
        mul_tensor_4: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(arg164_1, sub_tensor_1);  arg164_1 = sub_tensor_1 = None
        mul_tensor_5: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_tensor, arg60_1);  arg60_1 = None
        sum_dim_int_list_2: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_5, [0, 1]);  mul_tensor_5 = None
        sum_dim_int_list_3: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_tensor, [0, 1]);  add_tensor = None
        select_scatter_default: "f32[32, 512, 768, 2]" = torch.ops.aten.select_scatter.default(full_2, mul_tensor_4, 3, 0);  full_2 = mul_tensor_4 = None
        return (sum_dim_int_list_2, sum_dim_int_list_3, select_scatter_default)



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
