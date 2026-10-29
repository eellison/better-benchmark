"""Curated MM repro. Pattern hash: f7087915edf4."""
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
    def forward(self, arg0_1: "f32[768, 768]", arg1_1: "bf16[8192, 768]", arg2_1: "bf16[768]", arg3_1: "b8[8, 1024, 768]", _shape_param_0):
        # No stacktrace found for following nodes
        convert_element_type: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.bfloat16);  arg0_1 = None
        mm: "bf16[8192, 768]" = torch.ops.aten.mm.default(arg1_1, convert_element_type);  arg1_1 = convert_element_type = None
        view: "bf16[8, 1024, 768]" = torch.ops.aten.view.default(mm, _shape_param_0);  mm = _shape_param_0 = None
        add: "bf16[8, 1024, 768]" = torch.ops.aten.add.Tensor(view, arg2_1);  view = arg2_1 = None
        mul: "bf16[8, 1024, 768]" = torch.ops.aten.mul.Tensor(arg3_1, add);  arg3_1 = add = None
        mul_1: "bf16[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul, 1.1111111111111112);  mul = None
        return mul_1

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
