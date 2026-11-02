"""Curated MM repro. Pattern hash: 72c2a1c98651."""
import sys
from pathlib import Path
from math import inf, nan
import torch
import torch._inductor.inductor_prims
from torch import device
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from repro_harness import benchmark_repro, load_shape_configs, make_inputs_from_config
from mm_replay import make_inputs_from_config, benchmark_repro_safe as benchmark_repro
_repro_version = 3

class Repro(torch.nn.Module):
    def forward(self, arg0_1: "bf16[401408, 384]", arg1_1: "bf16[384, 128]", arg2_1: "f32[128]", arg3_1: "bf16[128, 128, 56, 56]", arg4_1: "f32[128, 56, 56, 1]", arg5_1: "f32[128, 56, 56, 1]", arg6_1: "f32[128]", arg7_1: "f32[128]", arg8_1: "f32[128, 56, 56, 1]", arg9_1: "f32[128, 56, 56, 1]", arg10_1: "f32[128, 56, 56, 128]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # No stacktrace found for following nodes
        mm: "bf16[401408, 128]" = torch.ops.aten.mm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        view: "bf16[8192, 49, 128]" = torch.ops.aten.view.default(mm, _shape_param_0);  mm = _shape_param_0 = None
        convert_element_type: "f32[8192, 49, 128]" = torch.ops.prims.convert_element_type.default(view, torch.float32);  view = None
        view_1: "f32[8192, 7, 7, 128]" = torch.ops.aten.view.default(convert_element_type, _shape_param_1);  convert_element_type = _shape_param_1 = None
        view_2: "f32[128, 8, 8, 7, 7, 128]" = torch.ops.aten.view.default(view_1, _shape_param_2);  view_1 = _shape_param_2 = None
        permute: "f32[128, 8, 7, 8, 7, 128]" = torch.ops.aten.permute.default(view_2, [0, 1, 3, 2, 4, 5]);  view_2 = None
        clone: "f32[128, 8, 7, 8, 7, 128]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format);  permute = None
        view_3: "f32[128, 56, 56, 128]" = torch.ops.aten.view.default(clone, _shape_param_3);  clone = _shape_param_3 = None
        mul: "f32[128, 56, 56, 128]" = torch.ops.aten.mul.Tensor(view_3, arg2_1);  arg2_1 = None
        mul_1: "f32[128, 56, 56, 128]" = torch.ops.aten.mul.Tensor(mul, 128)
        sum_1: "f32[128, 56, 56, 1]" = torch.ops.aten.sum.dim_IntList(mul, [3], True)
        permute_1: "bf16[128, 56, 56, 128]" = torch.ops.aten.permute.default(arg3_1, [0, 2, 3, 1]);  arg3_1 = None
        convert_element_type_1: "f32[128, 56, 56, 128]" = torch.ops.prims.convert_element_type.default(permute_1, torch.float32);  permute_1 = None
        sub: "f32[128, 56, 56, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_1, arg4_1);  convert_element_type_1 = arg4_1 = None
        mul_2: "f32[128, 56, 56, 128]" = torch.ops.aten.mul.Tensor(sub, arg5_1);  sub = None
        mul_3: "f32[128, 56, 56, 128]" = torch.ops.aten.mul.Tensor(mul_2, arg6_1)
        add: "f32[128, 56, 56, 128]" = torch.ops.aten.add.Tensor(mul_3, arg7_1);  mul_3 = arg7_1 = None
        sub_1: "f32[128, 56, 56, 128]" = torch.ops.aten.sub.Tensor(add, arg8_1);  add = arg8_1 = None
        mul_4: "f32[128, 56, 56, 128]" = torch.ops.aten.mul.Tensor(sub_1, arg9_1);  sub_1 = None
        mul_5: "f32[128, 56, 56, 128]" = torch.ops.aten.mul.Tensor(mul, mul_4);  mul = None
        sum_2: "f32[128, 56, 56, 1]" = torch.ops.aten.sum.dim_IntList(mul_5, [3], True);  mul_5 = None
        mul_6: "f32[128, 56, 56, 128]" = torch.ops.aten.mul.Tensor(mul_4, sum_2);  sum_2 = None
        sub_2: "f32[128, 56, 56, 128]" = torch.ops.aten.sub.Tensor(mul_1, sum_1);  mul_1 = sum_1 = None
        sub_3: "f32[128, 56, 56, 128]" = torch.ops.aten.sub.Tensor(sub_2, mul_6);  sub_2 = mul_6 = None
        div: "f32[128, 56, 56, 1]" = torch.ops.aten.div.Tensor(arg9_1, 128);  arg9_1 = None
        mul_7: "f32[128, 56, 56, 128]" = torch.ops.aten.mul.Tensor(div, sub_3);  div = sub_3 = None
        mul_8: "f32[128, 56, 56, 128]" = torch.ops.aten.mul.Tensor(view_3, mul_4);  mul_4 = None
        sum_3: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_8, [0, 1, 2]);  mul_8 = None
        sum_4: "f32[128]" = torch.ops.aten.sum.dim_IntList(view_3, [0, 1, 2]);  view_3 = None
        add_1: "f32[128, 56, 56, 128]" = torch.ops.aten.add.Tensor(arg10_1, mul_7);  arg10_1 = mul_7 = None
        mul_9: "f32[128, 56, 56, 128]" = torch.ops.aten.mul.Tensor(add_1, arg6_1);  arg6_1 = None
        mul_10: "f32[128, 56, 56, 128]" = torch.ops.aten.mul.Tensor(mul_9, 128)
        sum_5: "f32[128, 56, 56, 1]" = torch.ops.aten.sum.dim_IntList(mul_9, [3], True)
        mul_11: "f32[128, 56, 56, 128]" = torch.ops.aten.mul.Tensor(mul_9, mul_2);  mul_9 = None
        sum_6: "f32[128, 56, 56, 1]" = torch.ops.aten.sum.dim_IntList(mul_11, [3], True);  mul_11 = None
        mul_12: "f32[128, 56, 56, 128]" = torch.ops.aten.mul.Tensor(mul_2, sum_6);  sum_6 = None
        sub_4: "f32[128, 56, 56, 128]" = torch.ops.aten.sub.Tensor(mul_10, sum_5);  mul_10 = sum_5 = None
        sub_5: "f32[128, 56, 56, 128]" = torch.ops.aten.sub.Tensor(sub_4, mul_12);  sub_4 = mul_12 = None
        div_1: "f32[128, 56, 56, 1]" = torch.ops.aten.div.Tensor(arg5_1, 128);  arg5_1 = None
        mul_13: "f32[128, 56, 56, 128]" = torch.ops.aten.mul.Tensor(div_1, sub_5);  div_1 = sub_5 = None
        mul_14: "f32[128, 56, 56, 128]" = torch.ops.aten.mul.Tensor(add_1, mul_2);  mul_2 = None
        sum_7: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_14, [0, 1, 2]);  mul_14 = None
        sum_8: "f32[128]" = torch.ops.aten.sum.dim_IntList(add_1, [0, 1, 2]);  add_1 = None
        convert_element_type_2: "bf16[128, 56, 56, 128]" = torch.ops.prims.convert_element_type.default(mul_13, torch.bfloat16);  mul_13 = None
        permute_2: "bf16[128, 128, 56, 56]" = torch.ops.aten.permute.default(convert_element_type_2, [0, 3, 1, 2]);  convert_element_type_2 = None
        sum_9: "bf16[128]" = torch.ops.aten.sum.dim_IntList(permute_2, [0, 2, 3])
        convert_element_type_3: "f32[128]" = torch.ops.prims.convert_element_type.default(sum_9, torch.float32);  sum_9 = None
        return (sum_3, sum_4, sum_7, sum_8, permute_2, convert_element_type_3)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
