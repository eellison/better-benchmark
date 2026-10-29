"""Curated MM repro. Pattern hash: 2d123867f576."""
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
    def forward(self, arg0_1: "bf16[512, 4, 16, 64]", arg1_1: "bf16[240, 240]", arg2_1: "bf16[240]", arg3_1: "bf16[512, 16, 240]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        slice_1: "bf16[512, 4, 16, 60]" = torch.ops.aten.slice.Tensor(arg0_1, -1, 0, 60);  arg0_1 = None
        permute: "bf16[512, 16, 4, 60]" = torch.ops.aten.permute.default(slice_1, [0, 2, 1, 3]);  slice_1 = None
        clone: "bf16[512, 16, 4, 60]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format);  permute = None
        view_1: "bf16[8192, 240]" = torch.ops.aten.view.default(clone, _shape_param_0);  clone = _shape_param_0 = None
        addmm: "bf16[8192, 240]" = torch.ops.aten.addmm.default(arg2_1, view_1, arg1_1);  arg2_1 = view_1 = arg1_1 = None
        view_2: "bf16[512, 16, 240]" = torch.ops.aten.view.default(addmm, _shape_param_1);  addmm = _shape_param_1 = None
        add: "bf16[512, 16, 240]" = torch.ops.aten.add.Tensor(arg3_1, view_2);  arg3_1 = view_2 = None
        convert_element_type: "f32[512, 16, 240]" = torch.ops.prims.convert_element_type.default(add, torch.float32)
        return (add, convert_element_type)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
