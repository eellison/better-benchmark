"""
Standalone repro captured via capture_hook.
Label: torchbench_nanogpt_train_001
Pattern hash: c0e210b86b78
Shape hash: 4e28ca44
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
_shapes_config = "(T([1, 50304], f32), T([1, 768], f32), T([64, 768], f32), T([768], f32), T([1, 64, 768], f32), T([1, 64, 1], f32), T([1, 64, 768], f32), T([1, 64], i64, gen=Index(1024)), T([1, 64], i64, gen=Index(50304)), S([1, 64, 768]))"

class Repro(torch.nn.Module):
    def forward(self, view: "f32[1, 50304]", arg234_1: "f32[1, 768]", mm_95: "f32[64, 768]", arg2_1: "f32[768]", arg76_1: "f32[1, 64, 768]", arg259_1: "f32[1, 64, 1]", add_70: "f32[1, 64, 768]", arg75_1: "i64[1, 64]", arg0_1: "i64[1, 64]", _shape_param_0):
        # No stacktrace found for following nodes
        permute_default: "f32[50304, 1]" = torch.ops.aten.permute.default(view, [1, 0]);  view = None
        mul_tensor: "f32[50304, 768]" = torch.ops.aten.mul.Tensor(permute_default, arg234_1);  permute_default = arg234_1 = None
        view_default: "f32[1, 64, 768]" = torch.ops.aten.view.default(mm_95, _shape_param_0);  mm_95 = _shape_param_0 = None
        mul_tensor_1: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(view_default, arg2_1);  arg2_1 = None
        mul_tensor_2: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_1, 768)
        sum_dim_int_list: "f32[1, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_1, [2], True)
        mul_tensor_3: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_1, arg76_1);  mul_tensor_1 = None
        sum_dim_int_list_1: "f32[1, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_3, [2], True);  mul_tensor_3 = None
        mul_tensor_4: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(arg76_1, sum_dim_int_list_1);  sum_dim_int_list_1 = None
        sub_tensor: "f32[1, 64, 768]" = torch.ops.aten.sub.Tensor(mul_tensor_2, sum_dim_int_list);  mul_tensor_2 = sum_dim_int_list = None
        sub_tensor_1: "f32[1, 64, 768]" = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor_4);  sub_tensor = mul_tensor_4 = None
        mul_tensor_5: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(arg259_1, sub_tensor_1);  arg259_1 = sub_tensor_1 = None
        mul_tensor_6: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(view_default, arg76_1);  arg76_1 = None
        sum_dim_int_list_2: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_6, [0, 1]);  mul_tensor_6 = None
        sum_dim_int_list_3: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_default, [0, 1]);  view_default = None
        add_tensor: "f32[1, 64, 768]" = torch.ops.aten.add.Tensor(add_70, mul_tensor_5);  add_70 = mul_tensor_5 = None
        eq_scalar: "b8[1, 64]" = torch.ops.aten.eq.Scalar(arg75_1, -1)
        unsqueeze_default: "b8[1, 64, 1]" = torch.ops.aten.unsqueeze.default(eq_scalar, -1);  eq_scalar = None
        full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "f32[1, 64, 768]" = torch.ops.aten.where.self(unsqueeze_default, full_default, add_tensor);  unsqueeze_default = None
        full_default_1: "f32[1024, 768]" = torch.ops.aten.full.default([1024, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default: "f32[1024, 768]" = torch.ops.aten.index_put.default(full_default_1, [arg75_1], where_self, True);  full_default_1 = arg75_1 = where_self = None
        eq_scalar_1: "b8[1, 64]" = torch.ops.aten.eq.Scalar(arg0_1, -1)
        unsqueeze_default_1: "b8[1, 64, 1]" = torch.ops.aten.unsqueeze.default(eq_scalar_1, -1);  eq_scalar_1 = None
        where_self_1: "f32[1, 64, 768]" = torch.ops.aten.where.self(unsqueeze_default_1, full_default, add_tensor);  unsqueeze_default_1 = full_default = add_tensor = None
        full_default_2: "f32[50304, 768]" = torch.ops.aten.full.default([50304, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default_1: "f32[50304, 768]" = torch.ops.aten.index_put.default(full_default_2, [arg0_1], where_self_1, True);  full_default_2 = arg0_1 = where_self_1 = None
        add_tensor_1: "f32[50304, 768]" = torch.ops.aten.add.Tensor(mul_tensor, index_put_default_1);  mul_tensor = index_put_default_1 = None
        return (sum_dim_int_list_2, sum_dim_int_list_3, index_put_default, add_tensor_1)



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
