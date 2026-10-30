"""Curated MM repro. Pattern hash: 2e72db2e9d86."""
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
    def forward(self, arg0_1: "bf16[128, 1000]", arg1_1: "bf16[1000, 3072]", arg2_1: "bf16[128, 3072, 6, 6]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5):
        # No stacktrace found for following nodes
        mm: "bf16[128, 3072]" = torch.ops.aten.mm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        full: "bf16[393216]" = torch.ops.aten.full.default(_shape_param_0, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_0 = None
        as_strided_scatter: "bf16[393216]" = torch.ops.aten.as_strided_scatter.default(full, mm, _shape_param_1, _shape_param_2, 0);  full = mm = _shape_param_1 = _shape_param_2 = None
        as_strided: "bf16[128, 3072, 1, 1]" = torch.ops.aten.as_strided.default(as_strided_scatter, _shape_param_3, _shape_param_4, 0);  as_strided_scatter = _shape_param_3 = _shape_param_4 = None
        expand: "bf16[128, 3072, 6, 6]" = torch.ops.aten.expand.default(as_strided, _shape_param_5);  as_strided = _shape_param_5 = None
        div: "bf16[128, 3072, 6, 6]" = torch.ops.aten.div.Scalar(expand, 36);  expand = None
        mul: "bf16[128, 3072, 6, 6]" = torch.ops.aten.mul.Tensor(div, 1.7015043497085571);  div = None
        convert_element_type: "f32[128, 3072, 6, 6]" = torch.ops.prims.convert_element_type.default(mul, torch.float32);  mul = None
        convert_element_type_1: "f32[128, 3072, 6, 6]" = torch.ops.prims.convert_element_type.default(arg2_1, torch.float32);  arg2_1 = None
        mul_1: "f32[128, 3072, 6, 6]" = torch.ops.aten.mul.Tensor(convert_element_type_1, 0.7071067811865476)
        erf: "f32[128, 3072, 6, 6]" = torch.ops.aten.erf.default(mul_1);  mul_1 = None
        add: "f32[128, 3072, 6, 6]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_2: "f32[128, 3072, 6, 6]" = torch.ops.aten.mul.Tensor(add, 0.5);  add = None
        mul_3: "f32[128, 3072, 6, 6]" = torch.ops.aten.mul.Tensor(convert_element_type_1, convert_element_type_1)
        mul_4: "f32[128, 3072, 6, 6]" = torch.ops.aten.mul.Tensor(mul_3, -0.5);  mul_3 = None
        exp: "f32[128, 3072, 6, 6]" = torch.ops.aten.exp.default(mul_4);  mul_4 = None
        mul_5: "f32[128, 3072, 6, 6]" = torch.ops.aten.mul.Tensor(exp, 0.3989422804014327);  exp = None
        mul_6: "f32[128, 3072, 6, 6]" = torch.ops.aten.mul.Tensor(convert_element_type_1, mul_5);  convert_element_type_1 = mul_5 = None
        add_1: "f32[128, 3072, 6, 6]" = torch.ops.aten.add.Tensor(mul_2, mul_6);  mul_2 = mul_6 = None
        mul_7: "f32[128, 3072, 6, 6]" = torch.ops.aten.mul.Tensor(convert_element_type, add_1);  convert_element_type = add_1 = None
        convert_element_type_2: "bf16[128, 3072, 6, 6]" = torch.ops.prims.convert_element_type.default(mul_7, torch.bfloat16);  mul_7 = None
        return convert_element_type_2

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
