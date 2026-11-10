"""Curated MM repro. Pattern hash: 02bb2512abc0."""
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
    def forward(self, arg0_1: "bf16[8, 1024, 16, 64]", arg1_1: "bf16[8, 1024, 16, 64]", arg2_1: "bf16[8192, 1024]", arg3_1: "bf16[8, 1024, 16, 64]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        view_2: "bf16[8192, 1024]" = torch.ops.aten.view.default(arg0_1, _shape_param_0);  arg0_1 = _shape_param_0 = None
        permute_2: "bf16[1024, 8192]" = torch.ops.aten.permute.default(view_2, [1, 0]);  view_2 = None
        mm: "bf16[1024, 1024]" = torch.ops.aten.mm.default(permute_2, arg2_1);  permute_2 = None
        convert_element_type: "f32[1024, 1024]" = torch.ops.prims.convert_element_type.default(mm, torch.float32);  mm = None
        view_3: "bf16[8192, 1024]" = torch.ops.aten.view.default(arg1_1, _shape_param_1);  arg1_1 = _shape_param_1 = None
        permute_3: "bf16[1024, 8192]" = torch.ops.aten.permute.default(view_3, [1, 0]);  view_3 = None
        mm_1: "bf16[1024, 1024]" = torch.ops.aten.mm.default(permute_3, arg2_1);  permute_3 = None
        convert_element_type_1: "f32[1024, 1024]" = torch.ops.prims.convert_element_type.default(mm_1, torch.float32);  mm_1 = None
        view_5: "bf16[8192, 1024]" = torch.ops.aten.view.default(arg3_1, _shape_param_2);  arg3_1 = _shape_param_2 = None
        permute_5: "bf16[1024, 8192]" = torch.ops.aten.permute.default(view_5, [1, 0]);  view_5 = None
        mm_2: "bf16[1024, 1024]" = torch.ops.aten.mm.default(permute_5, arg2_1);  permute_5 = arg2_1 = None
        convert_element_type_2: "f32[1024, 1024]" = torch.ops.prims.convert_element_type.default(mm_2, torch.float32);  mm_2 = None
        return (convert_element_type, convert_element_type_1, convert_element_type_2)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
