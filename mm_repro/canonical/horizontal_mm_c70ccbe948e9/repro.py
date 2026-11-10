"""Curated MM repro. Pattern hash: c70ccbe948e9."""
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
    def forward(self, arg0_1: "bf16[1024, 16384]", arg1_1: "bf16[16384, 1024]", arg2_1: "bf16[16384, 1024]", arg3_1: "bf16[16384, 1024]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        mm: "bf16[1024, 1024]" = torch.ops.aten.mm.default(arg0_1, arg1_1);  arg1_1 = None
        squeeze_1: "bf16[1024, 16, 64]" = torch.ops.aten.view.default(mm, _shape_param_0);  mm = _shape_param_0 = None
        convert_element_type: "f32[1024, 16, 64]" = torch.ops.prims.convert_element_type.default(squeeze_1, torch.float32);  squeeze_1 = None
        mm_1: "bf16[1024, 1024]" = torch.ops.aten.mm.default(arg0_1, arg2_1);  arg2_1 = None
        squeeze_3: "bf16[1024, 16, 64]" = torch.ops.aten.view.default(mm_1, _shape_param_1);  mm_1 = _shape_param_1 = None
        convert_element_type_1: "f32[1024, 16, 64]" = torch.ops.prims.convert_element_type.default(squeeze_3, torch.float32);  squeeze_3 = None
        mm_2: "bf16[1024, 1024]" = torch.ops.aten.mm.default(arg0_1, arg3_1);  arg0_1 = arg3_1 = None
        squeeze_5: "bf16[1024, 16, 64]" = torch.ops.aten.view.default(mm_2, _shape_param_2);  mm_2 = _shape_param_2 = None
        convert_element_type_2: "f32[1024, 16, 64]" = torch.ops.prims.convert_element_type.default(squeeze_5, torch.float32);  squeeze_5 = None
        return (convert_element_type, convert_element_type_1, convert_element_type_2)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
