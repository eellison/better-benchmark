"""
Standalone repro captured via capture_hook.
Label: torchbench_nanogpt_train_001
Pattern hash: 72f8a586759c
Shape hash: 3b21f4a4
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([1, 768], f32), T([1], i64, gen=Index(1)), T([768], f32), T([1, 64, 768], f32), T([1, 64, 1], f32), S([1, 1, 768]), S([64, 768]), S([768]))"

class Repro(torch.nn.Module):
    def forward(self, mm: "f32[1, 768]", arg233_1: "i64[1]", arg74_1: "f32[768]", arg232_1: "f32[1, 64, 768]", arg235_1: "f32[1, 64, 1]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        view_default: "f32[1, 1, 768]" = torch.ops.aten.view.default(mm, _shape_param_0);  mm = _shape_param_0 = None
        full_default: "f32[1, 64, 768]" = torch.ops.aten.full.default([1, 64, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default: "f32[1, 64, 768]" = torch.ops.aten.index_put.default(full_default, [None, arg233_1], view_default, True);  full_default = arg233_1 = view_default = None
        mul_tensor: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(index_put_default, arg74_1);  arg74_1 = None
        mul_tensor_1: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, 768)
        sum_dim_int_list: "f32[1, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [2], True)
        mul_tensor_2: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, arg232_1);  mul_tensor = None
        sum_dim_int_list_1: "f32[1, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [2], True);  mul_tensor_2 = None
        mul_tensor_3: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(arg232_1, sum_dim_int_list_1);  sum_dim_int_list_1 = None
        sub_tensor: "f32[1, 64, 768]" = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_1: "f32[1, 64, 768]" = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor_3);  sub_tensor = mul_tensor_3 = None
        mul_tensor_4: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(arg235_1, sub_tensor_1);  arg235_1 = sub_tensor_1 = None
        mul_tensor_5: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(index_put_default, arg232_1);  arg232_1 = None
        sum_dim_int_list_2: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_5, [0, 1]);  mul_tensor_5 = None
        sum_dim_int_list_3: "f32[768]" = torch.ops.aten.sum.dim_IntList(index_put_default, [0, 1]);  index_put_default = None
        view_default_1: "f32[64, 768]" = torch.ops.aten.view.default(mul_tensor_4, _shape_param_1);  mul_tensor_4 = _shape_param_1 = None
        permute_default: "f32[768, 64]" = torch.ops.aten.permute.default(view_default_1, [1, 0])
        sum_dim_int_list_4: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_default_1, [0], True);  view_default_1 = None
        view_default_2: "f32[768]" = torch.ops.aten.view.default(sum_dim_int_list_4, _shape_param_2);  sum_dim_int_list_4 = _shape_param_2 = None
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
