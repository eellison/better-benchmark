"""
Standalone repro captured via capture_hook.
Label: torchbench_hf_GPT2_large_train_002
Pattern hash: 280231331614
Shape hash: 634af270
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
_shapes_config = "(T([2048, 1280], f32), T([1280], f32), T([4, 512, 1280], f32), T([4, 512, 1], f32), T([4, 512, 1280], f32), T([4, 512, 1280], b8), T([1, 512], i64, gen=Index(1024)), T([4, 512], i64, gen=Index(50257)), T([50257, 1280], f32), S([4, 512, 1280]))"

class Repro(torch.nn.Module):
    def forward(self, mm_288: "f32[2048, 1280]", arg2_1: "f32[1280]", arg221_1: "f32[4, 512, 1280]", arg836_1: "f32[4, 512, 1]", add_215: "f32[4, 512, 1280]", arg220_1: "b8[4, 512, 1280]", arg219_1: "i64[1, 512]", arg0_1: "i64[4, 512]", mm: "f32[50257, 1280]", _shape_param_0):
        # No stacktrace found for following nodes
        view_default: "f32[4, 512, 1280]" = torch.ops.aten.view.default(mm_288, _shape_param_0);  mm_288 = _shape_param_0 = None
        mul_tensor: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(view_default, arg2_1);  arg2_1 = None
        mul_tensor_1: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_tensor, 1280)
        sum_dim_int_list: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [2], True)
        mul_tensor_2: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_tensor, arg221_1);  mul_tensor = None
        sum_dim_int_list_1: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [2], True);  mul_tensor_2 = None
        mul_tensor_3: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(arg221_1, sum_dim_int_list_1);  sum_dim_int_list_1 = None
        sub_tensor: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_1: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor_3);  sub_tensor = mul_tensor_3 = None
        mul_tensor_4: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(arg836_1, sub_tensor_1);  arg836_1 = sub_tensor_1 = None
        mul_tensor_5: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(view_default, arg221_1);  arg221_1 = None
        sum_dim_int_list_2: "f32[1280]" = torch.ops.aten.sum.dim_IntList(mul_tensor_5, [0, 1]);  mul_tensor_5 = None
        sum_dim_int_list_3: "f32[1280]" = torch.ops.aten.sum.dim_IntList(view_default, [0, 1]);  view_default = None
        add_tensor: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(add_215, mul_tensor_4);  add_215 = mul_tensor_4 = None
        convert_element_type_default: "f32[4, 512, 1280]" = torch.ops.prims.convert_element_type.default(arg220_1, torch.float32);  arg220_1 = None
        mul_tensor_6: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 1.1111111111111112);  convert_element_type_default = None
        mul_tensor_7: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(add_tensor, mul_tensor_6);  add_tensor = mul_tensor_6 = None
        sum_dim_int_list_4: "f32[1, 512, 1280]" = torch.ops.aten.sum.dim_IntList(mul_tensor_7, [0], True)
        eq_scalar: "b8[1, 512]" = torch.ops.aten.eq.Scalar(arg219_1, -1)
        unsqueeze_default: "b8[1, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_scalar, -1);  eq_scalar = None
        full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "f32[1, 512, 1280]" = torch.ops.aten.where.self(unsqueeze_default, full_default, sum_dim_int_list_4);  unsqueeze_default = sum_dim_int_list_4 = None
        full_default_1: "f32[1024, 1280]" = torch.ops.aten.full.default([1024, 1280], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default: "f32[1024, 1280]" = torch.ops.aten.index_put.default(full_default_1, [arg219_1], where_self, True);  full_default_1 = arg219_1 = where_self = None
        eq_scalar_1: "b8[4, 512]" = torch.ops.aten.eq.Scalar(arg0_1, -1)
        unsqueeze_default_1: "b8[4, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_scalar_1, -1);  eq_scalar_1 = None
        where_self_1: "f32[4, 512, 1280]" = torch.ops.aten.where.self(unsqueeze_default_1, full_default, mul_tensor_7);  unsqueeze_default_1 = full_default = mul_tensor_7 = None
        full_default_2: "f32[50257, 1280]" = torch.ops.aten.full.default([50257, 1280], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default_1: "f32[50257, 1280]" = torch.ops.aten.index_put.default(full_default_2, [arg0_1], where_self_1, True);  full_default_2 = arg0_1 = where_self_1 = None
        add_tensor_1: "f32[50257, 1280]" = torch.ops.aten.add.Tensor(mm, index_put_default_1);  mm = index_put_default_1 = None
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
