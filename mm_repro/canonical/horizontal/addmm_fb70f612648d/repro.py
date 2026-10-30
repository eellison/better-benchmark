"""Curated MM repro. Pattern hash: fb70f612648d."""
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
    def forward(self, arg0_1: "f32[512, 128]", arg1_1: "bf16[128]", arg2_1: "bf16[32768, 512]", arg3_1: "f32[512, 128]", arg4_1: "bf16[128]", arg5_1: "f32[512, 128]", arg6_1: "bf16[128]", _shape_param_0):
        # No stacktrace found for following nodes
        convert_element_type: "bf16[512, 128]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.bfloat16);  arg0_1 = None
        addmm: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg1_1, arg2_1, convert_element_type);  arg1_1 = convert_element_type = None
        view: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm, _shape_param_0);  _shape_param_0 = None
        convert_element_type_1: "bf16[512, 128]" = torch.ops.prims.convert_element_type.default(arg3_1, torch.bfloat16);  arg3_1 = None
        addmm_1: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg4_1, arg2_1, convert_element_type_1);  arg4_1 = convert_element_type_1 = None
        convert_element_type_2: "bf16[512, 128]" = torch.ops.prims.convert_element_type.default(arg5_1, torch.bfloat16);  arg5_1 = None
        addmm_2: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg6_1, arg2_1, convert_element_type_2);  arg6_1 = arg2_1 = convert_element_type_2 = None
        return (addmm, view, addmm_1, addmm_2)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
