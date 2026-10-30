"""Curated MM repro. Pattern hash: 48674c069322."""
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
    def forward(self, arg0_1: "f32[8, 1024, 768]", arg1_1: "f32[8, 1024, 768]", arg2_1: "f32[8, 1024, 1]", arg3_1: "f32[8, 1024, 1]", arg4_1: "f32[8, 1024, 1]", arg5_1: "b8[8, 1024, 768]", arg6_1: "bf16[768, 3072]", arg7_1: "bf16[8192, 3072]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # No stacktrace found for following nodes
        mul: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(arg0_1, 768);  arg0_1 = None
        mul_1: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(arg1_1, arg2_1);  arg1_1 = arg2_1 = None
        sub: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(mul, arg3_1);  mul = arg3_1 = None
        sub_1: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(sub, mul_1);  sub = mul_1 = None
        mul_2: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(arg4_1, sub_1);  arg4_1 = sub_1 = None
        convert_element_type: "bf16[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(mul_2, torch.bfloat16);  mul_2 = None
        convert_element_type_1: "bf16[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(arg5_1, torch.bfloat16);  arg5_1 = None
        mul_3: "bf16[8, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_1, 1.1111111111111112);  convert_element_type_1 = None
        mul_4: "bf16[8, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type, mul_3);  convert_element_type = mul_3 = None
        view: "bf16[8192, 768]" = torch.ops.aten.view.default(mul_4, _shape_param_0);  mul_4 = _shape_param_0 = None
        mm: "bf16[8192, 3072]" = torch.ops.aten.mm.default(view, arg6_1);  view = arg6_1 = None
        view_1: "bf16[8, 1024, 3072]" = torch.ops.aten.view.default(mm, _shape_param_1);  mm = _shape_param_1 = None
        convert_element_type_2: "f32[8, 1024, 3072]" = torch.ops.prims.convert_element_type.default(view_1, torch.float32);  view_1 = None
        view_2: "bf16[8, 1024, 3072]" = torch.ops.aten.view.default(arg7_1, _shape_param_2);  arg7_1 = _shape_param_2 = None
        convert_element_type_3: "f32[8, 1024, 3072]" = torch.ops.prims.convert_element_type.default(view_2, torch.float32);  view_2 = None
        mul_5: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_3, 0.7071067811865476)
        erf: "f32[8, 1024, 3072]" = torch.ops.aten.erf.default(mul_5);  mul_5 = None
        add: "f32[8, 1024, 3072]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_6: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(add, 0.5);  add = None
        mul_7: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_3, convert_element_type_3)
        mul_8: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_7, -0.5);  mul_7 = None
        exp: "f32[8, 1024, 3072]" = torch.ops.aten.exp.default(mul_8);  mul_8 = None
        mul_9: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(exp, 0.3989422804014327);  exp = None
        mul_10: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_3, mul_9);  convert_element_type_3 = mul_9 = None
        add_1: "f32[8, 1024, 3072]" = torch.ops.aten.add.Tensor(mul_6, mul_10);  mul_6 = mul_10 = None
        mul_11: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_2, add_1);  convert_element_type_2 = add_1 = None
        convert_element_type_4: "bf16[8, 1024, 3072]" = torch.ops.prims.convert_element_type.default(mul_11, torch.bfloat16);  mul_11 = None
        view_3: "bf16[8192, 3072]" = torch.ops.aten.view.default(convert_element_type_4, _shape_param_3);  convert_element_type_4 = _shape_param_3 = None
        permute: "bf16[3072, 8192]" = torch.ops.aten.permute.default(view_3, [1, 0])
        return (view_3, permute)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
