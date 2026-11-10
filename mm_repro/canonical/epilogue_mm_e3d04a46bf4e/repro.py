"""Curated MM repro. Pattern hash: e3d04a46bf4e."""
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
    def forward(self, arg0_1: "bf16[8192, 1024]", arg1_1: "bf16[1024, 4096]", arg2_1: "b8[64, 128, 4096]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        mm: "bf16[8192, 4096]" = torch.ops.aten.mm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        view: "bf16[64, 128, 4096]" = torch.ops.aten.view.default(mm, _shape_param_0);  mm = _shape_param_0 = None
        full: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "bf16[64, 128, 4096]" = torch.ops.aten.where.self(arg2_1, full, view);  arg2_1 = full = view = None
        view_1: "bf16[8192, 4096]" = torch.ops.aten.view.default(where, _shape_param_1);  where = _shape_param_1 = None
        permute: "bf16[4096, 8192]" = torch.ops.aten.permute.default(view_1, [1, 0])
        sum_1: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1, [0], True, dtype = torch.float32)
        view_2: "f32[4096]" = torch.ops.aten.view.default(sum_1, _shape_param_2);  sum_1 = _shape_param_2 = None
        convert_element_type: "bf16[4096]" = torch.ops.prims.convert_element_type.default(view_2, torch.bfloat16);  view_2 = None
        convert_element_type_1: "f32[4096]" = torch.ops.prims.convert_element_type.default(convert_element_type, torch.float32);  convert_element_type = None
        return (view_1, permute, convert_element_type_1)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
