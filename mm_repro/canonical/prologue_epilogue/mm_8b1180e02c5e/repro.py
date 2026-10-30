"""Curated MM repro. Pattern hash: 8b1180e02c5e."""
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
    def forward(self, arg0_1: "f32[128, 1370, 768]", arg1_1: "f32[128, 1370, 768]", arg2_1: "f32[128, 1370, 1]", arg3_1: "f32[128, 1370, 1]", arg4_1: "f32[128, 1370, 1]", arg5_1: "f32[768]", arg6_1: "bf16[768, 3072]", arg7_1: "bf16[175360, 3072]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # No stacktrace found for following nodes
        mul: "f32[128, 1370, 768]" = torch.ops.aten.mul.Tensor(arg0_1, 768);  arg0_1 = None
        mul_1: "f32[128, 1370, 768]" = torch.ops.aten.mul.Tensor(arg1_1, arg2_1);  arg1_1 = arg2_1 = None
        sub: "f32[128, 1370, 768]" = torch.ops.aten.sub.Tensor(mul, arg3_1);  mul = arg3_1 = None
        sub_1: "f32[128, 1370, 768]" = torch.ops.aten.sub.Tensor(sub, mul_1);  sub = mul_1 = None
        mul_2: "f32[128, 1370, 768]" = torch.ops.aten.mul.Tensor(arg4_1, sub_1);  arg4_1 = sub_1 = None
        mul_3: "f32[128, 1370, 768]" = torch.ops.aten.mul.Tensor(mul_2, arg5_1);  mul_2 = arg5_1 = None
        convert_element_type: "bf16[128, 1370, 768]" = torch.ops.prims.convert_element_type.default(mul_3, torch.bfloat16);  mul_3 = None
        view: "bf16[175360, 768]" = torch.ops.aten.view.default(convert_element_type, _shape_param_0);  convert_element_type = _shape_param_0 = None
        mm: "bf16[175360, 3072]" = torch.ops.aten.mm.default(view, arg6_1);  view = arg6_1 = None
        view_1: "bf16[128, 1370, 3072]" = torch.ops.aten.view.default(mm, _shape_param_1);  mm = _shape_param_1 = None
        convert_element_type_1: "f32[128, 1370, 3072]" = torch.ops.prims.convert_element_type.default(view_1, torch.float32);  view_1 = None
        view_2: "bf16[128, 1370, 3072]" = torch.ops.aten.view.default(arg7_1, _shape_param_2);  arg7_1 = _shape_param_2 = None
        convert_element_type_2: "f32[128, 1370, 3072]" = torch.ops.prims.convert_element_type.default(view_2, torch.float32);  view_2 = None
        mul_4: "f32[128, 1370, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_2, 0.7071067811865476)
        erf: "f32[128, 1370, 3072]" = torch.ops.aten.erf.default(mul_4);  mul_4 = None
        add: "f32[128, 1370, 3072]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_5: "f32[128, 1370, 3072]" = torch.ops.aten.mul.Tensor(add, 0.5);  add = None
        mul_6: "f32[128, 1370, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_2, convert_element_type_2)
        mul_7: "f32[128, 1370, 3072]" = torch.ops.aten.mul.Tensor(mul_6, -0.5);  mul_6 = None
        exp: "f32[128, 1370, 3072]" = torch.ops.aten.exp.default(mul_7);  mul_7 = None
        mul_8: "f32[128, 1370, 3072]" = torch.ops.aten.mul.Tensor(exp, 0.3989422804014327);  exp = None
        mul_9: "f32[128, 1370, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_2, mul_8);  convert_element_type_2 = mul_8 = None
        add_1: "f32[128, 1370, 3072]" = torch.ops.aten.add.Tensor(mul_5, mul_9);  mul_5 = mul_9 = None
        mul_10: "f32[128, 1370, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_1, add_1);  convert_element_type_1 = add_1 = None
        convert_element_type_3: "bf16[128, 1370, 3072]" = torch.ops.prims.convert_element_type.default(mul_10, torch.bfloat16);  mul_10 = None
        view_3: "bf16[175360, 3072]" = torch.ops.aten.view.default(convert_element_type_3, _shape_param_3);  convert_element_type_3 = _shape_param_3 = None
        permute: "bf16[3072, 175360]" = torch.ops.aten.permute.default(view_3, [1, 0])
        return (view_3, permute)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
