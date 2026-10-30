"""Curated MM repro. Pattern hash: 12665fff5398."""
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
    def forward(self, arg0_1: "bf16[2048, 3072]", arg1_1: "bf16[3072, 768]", arg2_1: "f32[768]", arg3_1: "f32[16, 128, 768]", arg4_1: "f32[16, 128, 1]", _shape_param_0):
        # No stacktrace found for following nodes
        mm: "bf16[2048, 768]" = torch.ops.aten.mm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        view: "bf16[16, 128, 768]" = torch.ops.aten.view.default(mm, _shape_param_0);  mm = _shape_param_0 = None
        convert_element_type: "f32[16, 128, 768]" = torch.ops.prims.convert_element_type.default(view, torch.float32);  view = None
        mul: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(arg2_1, arg3_1)
        add: "f32[16, 128, 1]" = torch.ops.aten.add.Tensor(arg4_1, 1e-06);  arg4_1 = None
        div: "f32[16, 128, 768]" = torch.ops.aten.div.Tensor(mul, add);  mul = None
        div_1: "f32[16, 128, 768]" = torch.ops.aten.div.Tensor(div, add);  div = None
        neg: "f32[16, 128, 768]" = torch.ops.aten.neg.default(convert_element_type)
        mul_1: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(neg, div_1);  neg = div_1 = None
        div_2: "f32[16, 128, 768]" = torch.ops.aten.div.Tensor(convert_element_type, add);  add = None
        mul_2: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(div_2, arg2_1);  arg2_1 = None
        mul_3: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(div_2, arg3_1);  div_2 = arg3_1 = None
        return (convert_element_type, mul_1, mul_2, mul_3)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
