"""
Standalone repro captured via capture_hook.
Label: hf_MobileBertForMaskedLM_train_014
Pattern hash: 83ddb84cc830
Shape hash: 44c40231
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([32768, 512], f32), T([256, 128, 512], f32), T([32768, 512], f32), T([32768, 512], f32), T([256, 128, 512], f32), T([512], f32), T([], f32), T([256, 128], i64, gen=Index(2)), T([1, 512], i64, gen=Index(512)), S([256, 128, 512]), S([256, 128, 512]), S([256, 128, 512]), S([512]), S([512]), S([32768, 512]), S([512]))"

class Repro(torch.nn.Module):
    def forward(self, mm_714: "f32[32768, 512]", mul_516: "f32[256, 128, 512]", mm_720: "f32[32768, 512]", mm_722: "f32[32768, 512]", arg585_1: "f32[256, 128, 512]", arg3_1: "f32[512]", full_1: "f32[]", arg583_1: "i64[256, 128]", arg1_1: "i64[1, 512]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6):
        # No stacktrace found for following nodes
        view_default: "f32[256, 128, 512]" = torch.ops.aten.view.default(mm_714, _shape_param_0);  mm_714 = _shape_param_0 = None
        add_tensor: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(mul_516, view_default);  mul_516 = view_default = None
        view_default_1: "f32[256, 128, 512]" = torch.ops.aten.view.default(mm_720, _shape_param_1);  mm_720 = _shape_param_1 = None
        add_tensor_1: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(add_tensor, view_default_1);  add_tensor = view_default_1 = None
        view_default_2: "f32[256, 128, 512]" = torch.ops.aten.view.default(mm_722, _shape_param_2);  mm_722 = _shape_param_2 = None
        add_tensor_2: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(add_tensor_1, view_default_2);  add_tensor_1 = view_default_2 = None
        sum_dim_int_list: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(add_tensor_2, [0, 1], True)
        view_default_3: "f32[512]" = torch.ops.aten.view.default(sum_dim_int_list, _shape_param_3);  sum_dim_int_list = _shape_param_3 = None
        mul_tensor: "f32[256, 128, 512]" = torch.ops.aten.mul.Tensor(add_tensor_2, arg585_1);  arg585_1 = None
        mul_tensor_1: "f32[256, 128, 512]" = torch.ops.aten.mul.Tensor(add_tensor_2, arg3_1);  add_tensor_2 = arg3_1 = None
        sum_dim_int_list_1: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [0, 1], True);  mul_tensor = None
        view_default_4: "f32[512]" = torch.ops.aten.view.default(sum_dim_int_list_1, _shape_param_4);  sum_dim_int_list_1 = _shape_param_4 = None
        sum_dim_int_list_2: "f32[1, 128, 512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_1, [0], True)
        full_default: "b8[256, 128, 1]" = torch.ops.aten.full.default([256, 128, 1], False, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "f32[256, 128, 512]" = torch.ops.aten.where.self(full_default, full_1, mul_tensor_1);  full_default = None
        full_default_1: "f32[2, 512]" = torch.ops.aten.full.default([2, 512], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default: "f32[2, 512]" = torch.ops.aten.index_put.default(full_default_1, [arg583_1], where_self, True);  full_default_1 = arg583_1 = where_self = None
        slice_tensor: "i64[1, 128]" = torch.ops.aten.slice.Tensor(arg1_1, 1, 0, 128);  arg1_1 = None
        eq_scalar: "b8[1, 128]" = torch.ops.aten.eq.Scalar(slice_tensor, -1)
        unsqueeze_default: "b8[1, 128, 1]" = torch.ops.aten.unsqueeze.default(eq_scalar, -1);  eq_scalar = None
        where_self_1: "f32[1, 128, 512]" = torch.ops.aten.where.self(unsqueeze_default, full_1, sum_dim_int_list_2);  unsqueeze_default = full_1 = sum_dim_int_list_2 = None
        full_default_2: "f32[512, 512]" = torch.ops.aten.full.default([512, 512], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default_1: "f32[512, 512]" = torch.ops.aten.index_put.default(full_default_2, [slice_tensor], where_self_1, True);  full_default_2 = slice_tensor = where_self_1 = None
        view_default_5: "f32[32768, 512]" = torch.ops.aten.view.default(mul_tensor_1, _shape_param_5);  mul_tensor_1 = _shape_param_5 = None
        permute_default: "f32[512, 32768]" = torch.ops.aten.permute.default(view_default_5, [1, 0])
        sum_dim_int_list_3: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_default_5, [0], True);  view_default_5 = None
        view_default_6: "f32[512]" = torch.ops.aten.view.default(sum_dim_int_list_3, _shape_param_6);  sum_dim_int_list_3 = _shape_param_6 = None
        return (view_default_3, view_default_4, index_put_default, index_put_default_1, permute_default, view_default_6)

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
