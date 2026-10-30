"""Curated MM repro. Pattern hash: ca95d733844e."""
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
    def forward(self, arg0_1: "bf16[8192, 768]", arg1_1: "bf16[768, 3072]", arg2_1: "bf16[8192, 3072]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        mm: "bf16[8192, 3072]" = torch.ops.aten.mm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        view: "bf16[8, 1024, 3072]" = torch.ops.aten.view.default(mm, _shape_param_0);  mm = _shape_param_0 = None
        convert_element_type: "f32[8, 1024, 3072]" = torch.ops.prims.convert_element_type.default(view, torch.float32);  view = None
        view_1: "bf16[8, 1024, 3072]" = torch.ops.aten.view.default(arg2_1, _shape_param_1);  arg2_1 = _shape_param_1 = None
        convert_element_type_1: "f32[8, 1024, 3072]" = torch.ops.prims.convert_element_type.default(view_1, torch.float32);  view_1 = None
        mul: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_1, 0.7071067811865476)
        erf: "f32[8, 1024, 3072]" = torch.ops.aten.erf.default(mul);  mul = None
        add: "f32[8, 1024, 3072]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_1: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(add, 0.5);  add = None
        mul_2: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_1, convert_element_type_1)
        mul_3: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_2, -0.5);  mul_2 = None
        exp: "f32[8, 1024, 3072]" = torch.ops.aten.exp.default(mul_3);  mul_3 = None
        mul_4: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(exp, 0.3989422804014327);  exp = None
        mul_5: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_1, mul_4);  convert_element_type_1 = mul_4 = None
        add_1: "f32[8, 1024, 3072]" = torch.ops.aten.add.Tensor(mul_1, mul_5);  mul_1 = mul_5 = None
        mul_6: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type, add_1);  convert_element_type = add_1 = None
        convert_element_type_2: "bf16[8, 1024, 3072]" = torch.ops.prims.convert_element_type.default(mul_6, torch.bfloat16);  mul_6 = None
        view_2: "bf16[8192, 3072]" = torch.ops.aten.view.default(convert_element_type_2, _shape_param_2);  convert_element_type_2 = _shape_param_2 = None
        permute: "bf16[3072, 8192]" = torch.ops.aten.permute.default(view_2, [1, 0])
        return (view_2, permute)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
