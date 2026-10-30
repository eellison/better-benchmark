"""Curated MM repro. Pattern hash: 3759ef405a77."""
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
    def forward(self, arg0_1: "bf16[16, 128, 128]", arg1_1: "bf16[16, 128, 256]", arg2_1: "bf16[128, 4096]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        bmm: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        view: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm, _shape_param_0);  bmm = _shape_param_0 = None
        permute: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(view, [0, 2, 1, 3]);  view = None
        clone: "bf16[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format);  permute = None
        view_2: "bf16[128, 4096]" = torch.ops.aten.view.default(clone, _shape_param_1);  clone = _shape_param_1 = None
        permute_1: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_2, [1, 0])
        mm: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_1, arg2_1);  permute_1 = arg2_1 = None
        convert_element_type: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm, torch.float32);  mm = None
        return (view_2, convert_element_type)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
