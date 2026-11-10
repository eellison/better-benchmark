"""Curated MM repro. Pattern hash: 48c3e30370d8."""
import sys
from pathlib import Path
from math import inf, nan
import torch
import torch._inductor.inductor_prims
from torch import device
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from repro_harness import benchmark_repro, load_shape_configs, make_inputs_from_config
_repro_version = 3

class Repro(torch.nn.Module):
    def forward(self, arg0_1: "bf16[32768, 1024]", arg1_1: "bf16[1024, 256]", arg2_1: "f32[64, 512, 256]", arg3_1: "f32[256]", arg4_1: "bf16[64, 512, 256]", arg5_1: "f32[64, 512, 1]", arg6_1: "f32[64, 512, 1]", arg7_1: "b8[64, 512, 256]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        mm: "bf16[32768, 256]" = torch.ops.aten.mm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        view: "bf16[64, 512, 256]" = torch.ops.aten.view.default(mm, _shape_param_0);  mm = _shape_param_0 = None
        convert_element_type: "f32[64, 512, 256]" = torch.ops.prims.convert_element_type.default(view, torch.float32);  view = None
        add: "f32[64, 512, 256]" = torch.ops.aten.add.Tensor(arg2_1, convert_element_type);  arg2_1 = convert_element_type = None
        mul: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(add, arg3_1);  arg3_1 = None
        mul_1: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul, 256)
        sum_1: "f32[64, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul, [2], True)
        convert_element_type_1: "f32[64, 512, 256]" = torch.ops.prims.convert_element_type.default(arg4_1, torch.float32);  arg4_1 = None
        sub: "f32[64, 512, 256]" = torch.ops.aten.sub.Tensor(convert_element_type_1, arg5_1);  convert_element_type_1 = arg5_1 = None
        mul_2: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(sub, arg6_1);  sub = None
        mul_3: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul, mul_2);  mul = None
        sum_2: "f32[64, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_3, [2], True);  mul_3 = None
        mul_4: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_2, sum_2);  sum_2 = None
        sub_1: "f32[64, 512, 256]" = torch.ops.aten.sub.Tensor(mul_1, sum_1);  mul_1 = sum_1 = None
        sub_2: "f32[64, 512, 256]" = torch.ops.aten.sub.Tensor(sub_1, mul_4);  sub_1 = mul_4 = None
        div: "f32[64, 512, 1]" = torch.ops.aten.div.Tensor(arg6_1, 256);  arg6_1 = None
        mul_5: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(div, sub_2);  div = sub_2 = None
        mul_6: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(add, mul_2);  mul_2 = None
        sum_3: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_6, [0, 1]);  mul_6 = None
        sum_4: "f32[256]" = torch.ops.aten.sum.dim_IntList(add, [0, 1]);  add = None
        convert_element_type_2: "bf16[64, 512, 256]" = torch.ops.prims.convert_element_type.default(mul_5, torch.bfloat16);  mul_5 = None
        convert_element_type_3: "bf16[64, 512, 256]" = torch.ops.prims.convert_element_type.default(arg7_1, torch.bfloat16);  arg7_1 = None
        mul_7: "bf16[64, 512, 256]" = torch.ops.aten.mul.Tensor(convert_element_type_3, 1.1111111111111112);  convert_element_type_3 = None
        mul_8: "bf16[64, 512, 256]" = torch.ops.aten.mul.Tensor(convert_element_type_2, mul_7);  mul_7 = None
        view_1: "bf16[32768, 256]" = torch.ops.aten.view.default(mul_8, _shape_param_1);  mul_8 = _shape_param_1 = None
        permute: "bf16[256, 32768]" = torch.ops.aten.permute.default(view_1, [1, 0])
        sum_5: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_1, [0], True, dtype = torch.float32)
        view_2: "f32[256]" = torch.ops.aten.view.default(sum_5, _shape_param_2);  sum_5 = _shape_param_2 = None
        convert_element_type_4: "bf16[256]" = torch.ops.prims.convert_element_type.default(view_2, torch.bfloat16);  view_2 = None
        convert_element_type_5: "f32[256]" = torch.ops.prims.convert_element_type.default(convert_element_type_4, torch.float32);  convert_element_type_4 = None
        return (sum_3, sum_4, convert_element_type_2, view_1, permute, convert_element_type_5)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
