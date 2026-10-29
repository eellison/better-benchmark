"""Curated MM repro. Pattern hash: 052e9cb80a95."""
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
    def forward(self, arg0_1: "bf16[128, 3072, 8, 8]", arg1_1: "bf16[3072, 1000]", arg2_1: "bf16[1000]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        convert_element_type: "f32[128, 3072, 8, 8]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.float32);  arg0_1 = None
        mul: "f32[128, 3072, 8, 8]" = torch.ops.aten.mul.Tensor(convert_element_type, 0.5)
        mul_1: "f32[128, 3072, 8, 8]" = torch.ops.aten.mul.Tensor(convert_element_type, 0.7071067811865476);  convert_element_type = None
        erf: "f32[128, 3072, 8, 8]" = torch.ops.aten.erf.default(mul_1);  mul_1 = None
        add: "f32[128, 3072, 8, 8]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_2: "f32[128, 3072, 8, 8]" = torch.ops.aten.mul.Tensor(mul, add);  mul = add = None
        convert_element_type_1: "bf16[128, 3072, 8, 8]" = torch.ops.prims.convert_element_type.default(mul_2, torch.bfloat16);  mul_2 = None
        mul_3: "bf16[128, 3072, 8, 8]" = torch.ops.aten.mul.Tensor(convert_element_type_1, 1.7015043497085571);  convert_element_type_1 = None
        mean: "bf16[128, 3072, 1, 1]" = torch.ops.aten.mean.dim(mul_3, [2, 3], True);  mul_3 = None
        as_strided: "bf16[128, 3072, 1, 1]" = torch.ops.aten.as_strided.default(mean, _shape_param_0, _shape_param_1);  mean = _shape_param_0 = _shape_param_1 = None
        view: "bf16[128, 3072]" = torch.ops.aten.view.default(as_strided, _shape_param_2);  as_strided = _shape_param_2 = None
        addmm: "bf16[128, 1000]" = torch.ops.aten.addmm.default(arg2_1, view, arg1_1);  arg2_1 = view = arg1_1 = None
        return addmm

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
