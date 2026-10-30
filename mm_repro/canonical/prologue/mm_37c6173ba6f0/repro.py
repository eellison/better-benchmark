"""Curated MM repro. Pattern hash: 37c6173ba6f0."""
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
    def forward(self, arg0_1: "f32[32, 128, 2560]", arg1_1: "b8[32, 128, 2560]", arg2_1: "bf16[4096, 10240]", _shape_param_0):
        # No stacktrace found for following nodes
        convert_element_type: "bf16[32, 128, 2560]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.bfloat16);  arg0_1 = None
        convert_element_type_1: "bf16[32, 128, 2560]" = torch.ops.prims.convert_element_type.default(arg1_1, torch.bfloat16);  arg1_1 = None
        mul: "bf16[32, 128, 2560]" = torch.ops.aten.mul.Tensor(convert_element_type_1, 1.1111111111111112);  convert_element_type_1 = None
        mul_1: "bf16[32, 128, 2560]" = torch.ops.aten.mul.Tensor(convert_element_type, mul);  convert_element_type = mul = None
        view: "bf16[4096, 2560]" = torch.ops.aten.view.default(mul_1, _shape_param_0);  mul_1 = _shape_param_0 = None
        permute: "bf16[2560, 4096]" = torch.ops.aten.permute.default(view, [1, 0]);  view = None
        mm: "bf16[2560, 10240]" = torch.ops.aten.mm.default(permute, arg2_1);  permute = arg2_1 = None
        convert_element_type_2: "f32[2560, 10240]" = torch.ops.prims.convert_element_type.default(mm, torch.float32);  mm = None
        return convert_element_type_2

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
