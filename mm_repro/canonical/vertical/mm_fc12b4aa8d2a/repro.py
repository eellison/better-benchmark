"""Curated MM repro. Pattern hash: fc12b4aa8d2a."""
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
    def forward(self, arg0_1: "f32[64, 512, 128]", arg1_1: "f32[64, 512, 128]", arg2_1: "f32[64, 512, 128]", arg3_1: "f32[64, 512, 128]", arg4_1: "f32[64, 512, 1]", arg5_1: "f32[64, 512, 128]", arg6_1: "f32[64, 512, 128]", arg7_1: "bf16[128, 256]", arg8_1: "f32[256]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        sum_1: "f32[64, 512, 1]" = torch.ops.aten.sum.dim_IntList(arg0_1, [2], True);  arg0_1 = None
        sum_2: "f32[64, 512, 1]" = torch.ops.aten.sum.dim_IntList(arg1_1, [2], True);  arg1_1 = None
        mul: "f32[64, 512, 128]" = torch.ops.aten.mul.Tensor(arg2_1, sum_2);  arg2_1 = sum_2 = None
        sub: "f32[64, 512, 128]" = torch.ops.aten.sub.Tensor(arg3_1, sum_1);  arg3_1 = sum_1 = None
        sub_1: "f32[64, 512, 128]" = torch.ops.aten.sub.Tensor(sub, mul);  sub = mul = None
        div: "f32[64, 512, 1]" = torch.ops.aten.div.Tensor(arg4_1, 128);  arg4_1 = None
        mul_1: "f32[64, 512, 128]" = torch.ops.aten.mul.Tensor(div, sub_1);  div = sub_1 = None
        convert_element_type: "bf16[64, 512, 128]" = torch.ops.prims.convert_element_type.default(mul_1, torch.bfloat16);  mul_1 = None
        convert_element_type_1: "f32[64, 512, 128]" = torch.ops.prims.convert_element_type.default(convert_element_type, torch.float32);  convert_element_type = None
        mul_2: "f32[64, 512, 128]" = torch.ops.aten.mul.Tensor(arg5_1, 0.5);  arg5_1 = None
        mul_3: "f32[64, 512, 128]" = torch.ops.aten.mul.Tensor(arg6_1, arg6_1)
        mul_4: "f32[64, 512, 128]" = torch.ops.aten.mul.Tensor(mul_3, -0.5);  mul_3 = None
        exp: "f32[64, 512, 128]" = torch.ops.aten.exp.default(mul_4);  mul_4 = None
        mul_5: "f32[64, 512, 128]" = torch.ops.aten.mul.Tensor(exp, 0.3989422804014327);  exp = None
        mul_6: "f32[64, 512, 128]" = torch.ops.aten.mul.Tensor(arg6_1, mul_5);  arg6_1 = mul_5 = None
        add: "f32[64, 512, 128]" = torch.ops.aten.add.Tensor(mul_2, mul_6);  mul_2 = mul_6 = None
        mul_7: "f32[64, 512, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_1, add);  convert_element_type_1 = add = None
        convert_element_type_2: "bf16[64, 512, 128]" = torch.ops.prims.convert_element_type.default(mul_7, torch.bfloat16);  mul_7 = None
        view: "bf16[32768, 128]" = torch.ops.aten.view.default(convert_element_type_2, _shape_param_0);  convert_element_type_2 = _shape_param_0 = None
        mm: "bf16[32768, 256]" = torch.ops.aten.mm.default(view, arg7_1);  arg7_1 = None
        permute: "bf16[128, 32768]" = torch.ops.aten.permute.default(view, [1, 0])
        view_1: "bf16[64, 512, 256]" = torch.ops.aten.view.default(mm, _shape_param_1);  mm = _shape_param_1 = None
        convert_element_type_3: "f32[64, 512, 256]" = torch.ops.prims.convert_element_type.default(view_1, torch.float32);  view_1 = None
        mul_8: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(convert_element_type_3, arg8_1);  arg8_1 = None
        return (view, permute, convert_element_type_3, mul_8)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
