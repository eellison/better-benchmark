"""Curated MM repro. Pattern hash: 92913ace44ac."""
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
    def forward(self, arg0_1: "bf16[128, 768]", arg1_1: "bf16[768, 3072]", arg2_1: "bf16[128, 3072]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4):
        # No stacktrace found for following nodes
        full: "bf16[128, 1, 768]" = torch.ops.aten.full.default(_shape_param_0, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_0 = None
        select_scatter: "bf16[128, 1, 768]" = torch.ops.aten.select_scatter.default(full, arg0_1, 1, 0);  full = arg0_1 = None
        view: "bf16[128, 768]" = torch.ops.aten.view.default(select_scatter, _shape_param_1);  select_scatter = _shape_param_1 = None
        mm: "bf16[128, 3072]" = torch.ops.aten.mm.default(view, arg1_1);  view = arg1_1 = None
        view_1: "bf16[128, 1, 3072]" = torch.ops.aten.view.default(mm, _shape_param_2);  mm = _shape_param_2 = None
        convert_element_type: "f32[128, 1, 3072]" = torch.ops.prims.convert_element_type.default(view_1, torch.float32);  view_1 = None
        view_2: "bf16[128, 1, 3072]" = torch.ops.aten.view.default(arg2_1, _shape_param_3);  arg2_1 = _shape_param_3 = None
        convert_element_type_1: "f32[128, 1, 3072]" = torch.ops.prims.convert_element_type.default(view_2, torch.float32);  view_2 = None
        mul: "f32[128, 1, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_1, 0.7071067811865476)
        erf: "f32[128, 1, 3072]" = torch.ops.aten.erf.default(mul);  mul = None
        add: "f32[128, 1, 3072]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_1: "f32[128, 1, 3072]" = torch.ops.aten.mul.Tensor(add, 0.5);  add = None
        mul_2: "f32[128, 1, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_1, convert_element_type_1)
        mul_3: "f32[128, 1, 3072]" = torch.ops.aten.mul.Tensor(mul_2, -0.5);  mul_2 = None
        exp: "f32[128, 1, 3072]" = torch.ops.aten.exp.default(mul_3);  mul_3 = None
        mul_4: "f32[128, 1, 3072]" = torch.ops.aten.mul.Tensor(exp, 0.3989422804014327);  exp = None
        mul_5: "f32[128, 1, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_1, mul_4);  convert_element_type_1 = mul_4 = None
        add_1: "f32[128, 1, 3072]" = torch.ops.aten.add.Tensor(mul_1, mul_5);  mul_1 = mul_5 = None
        mul_6: "f32[128, 1, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type, add_1);  convert_element_type = add_1 = None
        convert_element_type_2: "bf16[128, 1, 3072]" = torch.ops.prims.convert_element_type.default(mul_6, torch.bfloat16);  mul_6 = None
        view_3: "bf16[128, 3072]" = torch.ops.aten.view.default(convert_element_type_2, _shape_param_4);  convert_element_type_2 = _shape_param_4 = None
        permute: "bf16[3072, 128]" = torch.ops.aten.permute.default(view_3, [1, 0])
        return (view_3, permute)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
