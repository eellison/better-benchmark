"""Curated MM repro. Pattern hash: 84f7dd0f25a6."""
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
    def forward(self, arg0_1: "bf16[32768, 256]", arg1_1: "bf16[256, 128]", arg2_1: "b8[64, 512, 128]", arg3_1: "f32[128]", arg4_1: "f32[64, 512, 128]", _shape_param_0):
        # No stacktrace found for following nodes
        mm: "bf16[32768, 128]" = torch.ops.aten.mm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        view: "bf16[64, 512, 128]" = torch.ops.aten.view.default(mm, _shape_param_0);  mm = _shape_param_0 = None
        convert_element_type: "f32[64, 512, 128]" = torch.ops.prims.convert_element_type.default(view, torch.float32);  view = None
        convert_element_type_1: "f32[64, 512, 128]" = torch.ops.prims.convert_element_type.default(arg2_1, torch.float32);  arg2_1 = None
        mul: "f32[64, 512, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_1, 1.1111111111111112);  convert_element_type_1 = None
        mul_1: "f32[64, 512, 128]" = torch.ops.aten.mul.Tensor(convert_element_type, mul);  convert_element_type = mul = None
        mul_2: "f32[64, 512, 128]" = torch.ops.aten.mul.Tensor(mul_1, arg3_1);  arg3_1 = None
        mul_3: "f32[64, 512, 128]" = torch.ops.aten.mul.Tensor(mul_2, 128)
        mul_4: "f32[64, 512, 128]" = torch.ops.aten.mul.Tensor(mul_1, arg4_1);  arg4_1 = None
        return (mul_1, mul_2, mul_3, mul_4)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
