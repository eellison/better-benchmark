"""Curated MM repro. Pattern hash: 0ebedcb77817."""
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
    def forward(self, arg0_1: "bf16[8192, 1024]", arg1_1: "bf16[1024, 4096]", arg2_1: "b8[512, 16, 4096]", arg3_1: "bf16[8192, 4096]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        mm: "bf16[8192, 4096]" = torch.ops.aten.mm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        view: "bf16[512, 16, 4096]" = torch.ops.aten.view.default(mm, _shape_param_0);  mm = _shape_param_0 = None
        convert_element_type: "bf16[512, 16, 4096]" = torch.ops.prims.convert_element_type.default(arg2_1, torch.bfloat16);  arg2_1 = None
        mul: "bf16[512, 16, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type, 1.1111111111111112);  convert_element_type = None
        mul_1: "bf16[512, 16, 4096]" = torch.ops.aten.mul.Tensor(view, mul);  view = mul = None
        convert_element_type_1: "f32[512, 16, 4096]" = torch.ops.prims.convert_element_type.default(mul_1, torch.float32);  mul_1 = None
        view_1: "bf16[512, 16, 4096]" = torch.ops.aten.view.default(arg3_1, _shape_param_1);  arg3_1 = _shape_param_1 = None
        convert_element_type_2: "f32[512, 16, 4096]" = torch.ops.prims.convert_element_type.default(view_1, torch.float32);  view_1 = None
        mul_2: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_2, 0.7071067811865476)
        erf: "f32[512, 16, 4096]" = torch.ops.aten.erf.default(mul_2);  mul_2 = None
        add: "f32[512, 16, 4096]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_3: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(add, 0.5);  add = None
        mul_4: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_2, convert_element_type_2)
        mul_5: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(mul_4, -0.5);  mul_4 = None
        exp: "f32[512, 16, 4096]" = torch.ops.aten.exp.default(mul_5);  mul_5 = None
        mul_6: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(exp, 0.3989422804014327);  exp = None
        mul_7: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_2, mul_6);  convert_element_type_2 = mul_6 = None
        add_1: "f32[512, 16, 4096]" = torch.ops.aten.add.Tensor(mul_3, mul_7);  mul_3 = mul_7 = None
        mul_8: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_1, add_1);  convert_element_type_1 = add_1 = None
        convert_element_type_3: "bf16[512, 16, 4096]" = torch.ops.prims.convert_element_type.default(mul_8, torch.bfloat16);  mul_8 = None
        view_2: "bf16[8192, 4096]" = torch.ops.aten.view.default(convert_element_type_3, _shape_param_2);  convert_element_type_3 = _shape_param_2 = None
        permute: "bf16[4096, 8192]" = torch.ops.aten.permute.default(view_2, [1, 0])
        return (view_2, permute)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
