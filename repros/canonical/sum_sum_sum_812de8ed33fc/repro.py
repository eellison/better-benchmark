"""
Standalone repro captured via capture_hook.
Label: torchbench_hf_Bert_large_train_001
Pattern hash: 812de8ed33fc
Shape hash: 4a252802
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
_shapes_config = "(T([2048, 1024], f32), T([4, 512, 1024], f32), T([2048, 1024], f32), T([2048, 1024], f32), T([4, 512, 1024], b8), T([1024], f32), T([4, 512, 1024], f32), T([4, 512, 1], f32), T([1, 512], i64, gen=Index(512)), T([], f32), T([1, 512], i64, gen=Index(2)), T([4, 512], i64, gen=Index(30522)), T([30522, 1024], f32), S([4, 512, 1024]), S([4, 512, 1024]), S([4, 512, 1024]), S([4, 512]))"

class Repro(torch.nn.Module):
    def forward(self, mm_286: "f32[2048, 1024]", mul_681: "f32[4, 512, 1024]", mm_288: "f32[2048, 1024]", mm_290: "f32[2048, 1024]", arg201_1: "b8[4, 512, 1024]", arg3_1: "f32[1024]", arg200_1: "f32[4, 512, 1024]", arg623_1: "f32[4, 512, 1]", arg1_1: "i64[1, 512]", full_1: "f32[]", arg199_1: "i64[1, 512]", arg0_1: "i64[4, 512]", mm_1: "f32[30522, 1024]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # No stacktrace found for following nodes
        view_default: "f32[4, 512, 1024]" = torch.ops.aten.view.default(mm_286, _shape_param_0);  mm_286 = _shape_param_0 = None
        add_tensor: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(mul_681, view_default);  mul_681 = view_default = None
        view_default_1: "f32[4, 512, 1024]" = torch.ops.aten.view.default(mm_288, _shape_param_1);  mm_288 = _shape_param_1 = None
        add_tensor_1: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_tensor, view_default_1);  add_tensor = view_default_1 = None
        view_default_2: "f32[4, 512, 1024]" = torch.ops.aten.view.default(mm_290, _shape_param_2);  mm_290 = _shape_param_2 = None
        add_tensor_2: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_tensor_1, view_default_2);  add_tensor_1 = view_default_2 = None
        convert_element_type_default: "f32[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(arg201_1, torch.float32);  arg201_1 = None
        mul_tensor: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 1.1111111111111112);  convert_element_type_default = None
        mul_tensor_1: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_tensor_2, mul_tensor);  add_tensor_2 = mul_tensor = None
        mul_tensor_2: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor_1, arg3_1);  arg3_1 = None
        mul_tensor_3: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor_2, 1024)
        sum_dim_int_list: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [2], True)
        mul_tensor_4: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor_2, arg200_1);  mul_tensor_2 = None
        sum_dim_int_list_1: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_4, [2], True);  mul_tensor_4 = None
        mul_tensor_5: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(arg200_1, sum_dim_int_list_1);  sum_dim_int_list_1 = None
        sub_tensor: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_tensor_3, sum_dim_int_list);  mul_tensor_3 = sum_dim_int_list = None
        sub_tensor_1: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor_5);  sub_tensor = mul_tensor_5 = None
        mul_tensor_6: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(arg623_1, sub_tensor_1);  arg623_1 = sub_tensor_1 = None
        mul_tensor_7: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor_1, arg200_1);  arg200_1 = None
        sum_dim_int_list_2: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_7, [0, 1]);  mul_tensor_7 = None
        sum_dim_int_list_3: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_1, [0, 1]);  mul_tensor_1 = None
        sum_dim_int_list_4: "f32[1, 512, 1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_6, [0], True)
        eq_scalar: "b8[1, 512]" = torch.ops.aten.eq.Scalar(arg1_1, -1)
        unsqueeze_default: "b8[1, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_scalar, -1);  eq_scalar = None
        where_self: "f32[1, 512, 1024]" = torch.ops.aten.where.self(unsqueeze_default, full_1, sum_dim_int_list_4);  unsqueeze_default = sum_dim_int_list_4 = None
        full_default: "f32[512, 1024]" = torch.ops.aten.full.default([512, 1024], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default: "f32[512, 1024]" = torch.ops.aten.index_put.default(full_default, [arg1_1], where_self, True);  full_default = arg1_1 = where_self = None
        expand_default: "i64[4, 512]" = torch.ops.aten.expand.default(arg199_1, _shape_param_3);  arg199_1 = _shape_param_3 = None
        eq_scalar_1: "b8[4, 512]" = torch.ops.aten.eq.Scalar(expand_default, -1)
        unsqueeze_default_1: "b8[4, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_scalar_1, -1);  eq_scalar_1 = None
        where_self_1: "f32[4, 512, 1024]" = torch.ops.aten.where.self(unsqueeze_default_1, full_1, mul_tensor_6);  unsqueeze_default_1 = None
        full_default_1: "f32[2, 1024]" = torch.ops.aten.full.default([2, 1024], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default_1: "f32[2, 1024]" = torch.ops.aten.index_put.default(full_default_1, [expand_default], where_self_1, True);  full_default_1 = expand_default = where_self_1 = None
        eq_scalar_2: "b8[4, 512]" = torch.ops.aten.eq.Scalar(arg0_1, 0)
        unsqueeze_default_2: "b8[4, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_scalar_2, -1);  eq_scalar_2 = None
        where_self_2: "f32[4, 512, 1024]" = torch.ops.aten.where.self(unsqueeze_default_2, full_1, mul_tensor_6);  unsqueeze_default_2 = full_1 = mul_tensor_6 = None
        full_default_2: "f32[30522, 1024]" = torch.ops.aten.full.default([30522, 1024], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default_2: "f32[30522, 1024]" = torch.ops.aten.index_put.default(full_default_2, [arg0_1], where_self_2, True);  full_default_2 = arg0_1 = where_self_2 = None
        add_tensor_3: "f32[30522, 1024]" = torch.ops.aten.add.Tensor(mm_1, index_put_default_2);  mm_1 = index_put_default_2 = None
        return (sum_dim_int_list_2, sum_dim_int_list_3, index_put_default, index_put_default_1, add_tensor_3)



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
