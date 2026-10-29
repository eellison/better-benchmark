"""Curated MM repro. Pattern hash: 90d35e6217f1."""
import sys
from pathlib import Path
from math import inf, nan
import torch
import torch._inductor.inductor_prims
from torch import device
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, load_shape_configs, make_inputs_from_config
_repro_version = 3

class Repro(torch.nn.Module):
    def forward(self, arg0_1: "f32[32, 128, 2560]", arg1_1: "b8[32, 128, 2560]", arg2_1: "bf16[2560, 10240]", arg3_1: "bf16[4096, 10240]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # No stacktrace found for following nodes
        convert_element_type: "bf16[32, 128, 2560]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.bfloat16);  arg0_1 = None
        convert_element_type_1: "bf16[32, 128, 2560]" = torch.ops.prims.convert_element_type.default(arg1_1, torch.bfloat16);  arg1_1 = None
        mul: "bf16[32, 128, 2560]" = torch.ops.aten.mul.Tensor(convert_element_type_1, 1.1111111111111112);  convert_element_type_1 = None
        mul_1: "bf16[32, 128, 2560]" = torch.ops.aten.mul.Tensor(convert_element_type, mul);  convert_element_type = mul = None
        view: "bf16[4096, 2560]" = torch.ops.aten.view.default(mul_1, _shape_param_0);  mul_1 = _shape_param_0 = None
        mm: "bf16[4096, 10240]" = torch.ops.aten.mm.default(view, arg2_1);  view = arg2_1 = None
        view_1: "bf16[32, 128, 10240]" = torch.ops.aten.view.default(mm, _shape_param_1);  mm = _shape_param_1 = None
        convert_element_type_2: "f32[32, 128, 10240]" = torch.ops.prims.convert_element_type.default(view_1, torch.float32);  view_1 = None
        view_2: "bf16[32, 128, 10240]" = torch.ops.aten.view.default(arg3_1, _shape_param_2);  arg3_1 = _shape_param_2 = None
        convert_element_type_3: "f32[32, 128, 10240]" = torch.ops.prims.convert_element_type.default(view_2, torch.float32);  view_2 = None
        mul_2: "f32[32, 128, 10240]" = torch.ops.aten.mul.Tensor(convert_element_type_3, 0.7071067811865476)
        erf: "f32[32, 128, 10240]" = torch.ops.aten.erf.default(mul_2);  mul_2 = None
        add: "f32[32, 128, 10240]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_3: "f32[32, 128, 10240]" = torch.ops.aten.mul.Tensor(add, 0.5);  add = None
        mul_4: "f32[32, 128, 10240]" = torch.ops.aten.mul.Tensor(convert_element_type_3, convert_element_type_3)
        mul_5: "f32[32, 128, 10240]" = torch.ops.aten.mul.Tensor(mul_4, -0.5);  mul_4 = None
        exp: "f32[32, 128, 10240]" = torch.ops.aten.exp.default(mul_5);  mul_5 = None
        mul_6: "f32[32, 128, 10240]" = torch.ops.aten.mul.Tensor(exp, 0.3989422804014327);  exp = None
        mul_7: "f32[32, 128, 10240]" = torch.ops.aten.mul.Tensor(convert_element_type_3, mul_6);  convert_element_type_3 = mul_6 = None
        add_1: "f32[32, 128, 10240]" = torch.ops.aten.add.Tensor(mul_3, mul_7);  mul_3 = mul_7 = None
        mul_8: "f32[32, 128, 10240]" = torch.ops.aten.mul.Tensor(convert_element_type_2, add_1);  convert_element_type_2 = add_1 = None
        convert_element_type_4: "bf16[32, 128, 10240]" = torch.ops.prims.convert_element_type.default(mul_8, torch.bfloat16);  mul_8 = None
        view_3: "bf16[4096, 10240]" = torch.ops.aten.view.default(convert_element_type_4, _shape_param_3);  convert_element_type_4 = _shape_param_3 = None
        permute: "bf16[10240, 4096]" = torch.ops.aten.permute.default(view_3, [1, 0])
        return (view_3, permute)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
