"""Curated MM repro. Pattern hash: ed374b42c928."""
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
    def forward(self, arg0_1: "bf16[16384, 384]", arg1_1: "bf16[384, 768]", arg2_1: "f32[32, 512, 768]", arg3_1: "bf16[16384, 768]", arg4_1: "bf16[32, 768, 512]", arg5_1: "bf16[16384, 768]", arg6_1: "bf16[16384, 768]", arg7_1: "f32[768]", arg8_1: "f32[32, 512, 768]", arg9_1: "f32[32, 512, 1]", arg10_1: "b8[32, 512, 768]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5):
        # No stacktrace found for following nodes
        mm: "bf16[16384, 768]" = torch.ops.aten.mm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        view: "bf16[32, 512, 768]" = torch.ops.aten.view.default(mm, _shape_param_0);  mm = _shape_param_0 = None
        convert_element_type: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(view, torch.float32);  view = None
        add: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(arg2_1, convert_element_type);  arg2_1 = convert_element_type = None
        view_1: "bf16[32, 512, 768]" = torch.ops.aten.view.default(arg3_1, _shape_param_1);  arg3_1 = _shape_param_1 = None
        convert_element_type_1: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(view_1, torch.float32);  view_1 = None
        add_1: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add, convert_element_type_1);  add = convert_element_type_1 = None
        convert_element_type_2: "f32[32, 768, 512]" = torch.ops.prims.convert_element_type.default(arg4_1, torch.float32);  arg4_1 = None
        permute: "f32[32, 512, 768]" = torch.ops.aten.permute.default(convert_element_type_2, [0, 2, 1]);  convert_element_type_2 = None
        add_2: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_1, permute);  add_1 = permute = None
        view_2: "bf16[32, 512, 768]" = torch.ops.aten.view.default(arg5_1, _shape_param_2);  arg5_1 = _shape_param_2 = None
        convert_element_type_3: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(view_2, torch.float32);  view_2 = None
        add_3: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_2, convert_element_type_3);  add_2 = convert_element_type_3 = None
        view_3: "bf16[32, 512, 768]" = torch.ops.aten.view.default(arg6_1, _shape_param_3);  arg6_1 = _shape_param_3 = None
        convert_element_type_4: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(view_3, torch.float32);  view_3 = None
        add_4: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_3, convert_element_type_4);  add_3 = convert_element_type_4 = None
        mul: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_4, arg7_1);  arg7_1 = None
        mul_1: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul, 768)
        sum_1: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul, [2], True)
        mul_2: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul, arg8_1);  mul = None
        sum_2: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_2, [2], True);  mul_2 = None
        mul_3: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(arg8_1, sum_2);  sum_2 = None
        sub: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_1, sum_1);  mul_1 = sum_1 = None
        sub_1: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub, mul_3);  sub = mul_3 = None
        mul_4: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(arg9_1, sub_1);  arg9_1 = sub_1 = None
        mul_5: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_4, arg8_1);  arg8_1 = None
        sum_3: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_5, [0, 1]);  mul_5 = None
        sum_4: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_4, [0, 1]);  add_4 = None
        convert_element_type_5: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(mul_4, torch.bfloat16)
        convert_element_type_6: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(arg10_1, torch.bfloat16);  arg10_1 = None
        mul_6: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_6, 1.1111111111111112);  convert_element_type_6 = None
        mul_7: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_5, mul_6);  convert_element_type_5 = mul_6 = None
        view_4: "bf16[16384, 768]" = torch.ops.aten.view.default(mul_7, _shape_param_4);  mul_7 = _shape_param_4 = None
        permute_1: "bf16[768, 16384]" = torch.ops.aten.permute.default(view_4, [1, 0])
        sum_5: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_4, [0], True, dtype = torch.float32)
        view_5: "f32[768]" = torch.ops.aten.view.default(sum_5, _shape_param_5);  sum_5 = _shape_param_5 = None
        convert_element_type_7: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_5, torch.bfloat16);  view_5 = None
        convert_element_type_8: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_7, torch.float32);  convert_element_type_7 = None
        return (mul_4, sum_3, sum_4, view_4, permute_1, convert_element_type_8)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
