"""
Standalone repro captured via capture_hook.
Label: hf_AlbertForMaskedLM_train_001
Pattern hash: 03d921e5910d
Shape hash: 16e32c20
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
_shapes_config = "(T([4096, 128], f32), T([128], f32), T([8, 512, 128], f32), T([8, 512, 1], f32), T([1, 512], i64, gen=Index(512)), T([], f32), T([1, 512], i64, gen=Index(2)), T([8, 512], i64, gen=Index(30000)), T([30000, 128], f32), S([8, 512, 128]), S([8, 512]))"

class Repro(torch.nn.Module):
    def forward(self, mm_148: "f32[4096, 128]", arg3_1: "f32[128]", arg17_1: "f32[8, 512, 128]", arg184_1: "f32[8, 512, 1]", arg1_1: "i64[1, 512]", full_1: "f32[]", arg16_1: "i64[1, 512]", arg0_1: "i64[8, 512]", mm_1: "f32[30000, 128]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        view_default: "f32[8, 512, 128]" = torch.ops.aten.view.default(mm_148, _shape_param_0);  mm_148 = _shape_param_0 = None
        mul_tensor: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(view_default, arg3_1);  arg3_1 = None
        mul_tensor_1: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(mul_tensor, 128)
        sum_dim_int_list: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [2], True)
        mul_tensor_2: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(mul_tensor, arg17_1);  mul_tensor = None
        sum_dim_int_list_1: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [2], True);  mul_tensor_2 = None
        mul_tensor_3: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(arg17_1, sum_dim_int_list_1);  sum_dim_int_list_1 = None
        sub_tensor: "f32[8, 512, 128]" = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_1: "f32[8, 512, 128]" = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor_3);  sub_tensor = mul_tensor_3 = None
        mul_tensor_4: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(arg184_1, sub_tensor_1);  arg184_1 = sub_tensor_1 = None
        mul_tensor_5: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(view_default, arg17_1);  arg17_1 = None
        sum_dim_int_list_2: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_tensor_5, [0, 1]);  mul_tensor_5 = None
        sum_dim_int_list_3: "f32[128]" = torch.ops.aten.sum.dim_IntList(view_default, [0, 1]);  view_default = None
        sum_dim_int_list_4: "f32[1, 512, 128]" = torch.ops.aten.sum.dim_IntList(mul_tensor_4, [0], True)
        eq_scalar: "b8[1, 512]" = torch.ops.aten.eq.Scalar(arg1_1, -1)
        unsqueeze_default: "b8[1, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_scalar, -1);  eq_scalar = None
        where_self: "f32[1, 512, 128]" = torch.ops.aten.where.self(unsqueeze_default, full_1, sum_dim_int_list_4);  unsqueeze_default = sum_dim_int_list_4 = None
        full_default: "f32[512, 128]" = torch.ops.aten.full.default([512, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default: "f32[512, 128]" = torch.ops.aten.index_put.default(full_default, [arg1_1], where_self, True);  full_default = arg1_1 = where_self = None
        expand_default: "i64[8, 512]" = torch.ops.aten.expand.default(arg16_1, _shape_param_1);  arg16_1 = _shape_param_1 = None
        eq_scalar_1: "b8[8, 512]" = torch.ops.aten.eq.Scalar(expand_default, -1)
        unsqueeze_default_1: "b8[8, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_scalar_1, -1);  eq_scalar_1 = None
        where_self_1: "f32[8, 512, 128]" = torch.ops.aten.where.self(unsqueeze_default_1, full_1, mul_tensor_4);  unsqueeze_default_1 = None
        full_default_1: "f32[2, 128]" = torch.ops.aten.full.default([2, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default_1: "f32[2, 128]" = torch.ops.aten.index_put.default(full_default_1, [expand_default], where_self_1, True);  full_default_1 = expand_default = where_self_1 = None
        eq_scalar_2: "b8[8, 512]" = torch.ops.aten.eq.Scalar(arg0_1, 0)
        unsqueeze_default_2: "b8[8, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_scalar_2, -1);  eq_scalar_2 = None
        where_self_2: "f32[8, 512, 128]" = torch.ops.aten.where.self(unsqueeze_default_2, full_1, mul_tensor_4);  unsqueeze_default_2 = full_1 = mul_tensor_4 = None
        full_default_2: "f32[30000, 128]" = torch.ops.aten.full.default([30000, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default_2: "f32[30000, 128]" = torch.ops.aten.index_put.default(full_default_2, [arg0_1], where_self_2, True);  full_default_2 = arg0_1 = where_self_2 = None
        add_tensor: "f32[30000, 128]" = torch.ops.aten.add.Tensor(mm_1, index_put_default_2);  mm_1 = index_put_default_2 = None
        return (sum_dim_int_list_3, sum_dim_int_list_2, index_put_default_1, add_tensor, index_put_default)



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
