"""Curated MM repro. Pattern hash: 547235d48e3f."""
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
    def forward(self, arg0_1: "bf16[8192, 3072]", arg1_1: "bf16[3072, 768]", arg2_1: "f32[8, 1024, 768]", arg3_1: "f32[768]", arg4_1: "f32[8, 1024, 768]", _shape_param_0):
        # No stacktrace found for following nodes
        mm: "bf16[8192, 768]" = torch.ops.aten.mm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        view: "bf16[8, 1024, 768]" = torch.ops.aten.view.default(mm, _shape_param_0);  mm = _shape_param_0 = None
        convert_element_type: "f32[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(view, torch.float32);  view = None
        add: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(arg2_1, convert_element_type);  arg2_1 = convert_element_type = None
        mul: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add, arg3_1);  arg3_1 = None
        mul_1: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul, 768)
        mul_2: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add, arg4_1);  arg4_1 = None
        return (add, mul, mul_1, mul_2)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
