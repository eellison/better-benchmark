"""Curated MM repro. Pattern hash: de42aa1b5558."""
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
    def forward(self, arg0_1: "bf16[16384, 1024]", arg1_1: "bf16[1024, 1024]", arg2_1: "bf16[1024, 64, 256]", arg3_1: "f32[1024, 256, 256]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        mm: "bf16[16384, 1024]" = torch.ops.aten.mm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        view_1: "bf16[64, 256, 16, 64]" = torch.ops.aten.view.default(mm, _shape_param_0);  mm = _shape_param_0 = None
        permute: "bf16[64, 16, 256, 64]" = torch.ops.aten.permute.default(view_1, [0, 2, 1, 3]);  view_1 = None
        clone: "bf16[64, 16, 256, 64]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format);  permute = None
        view_2: "bf16[1024, 256, 64]" = torch.ops.aten.view.default(clone, _shape_param_1);  clone = _shape_param_1 = None
        bmm: "bf16[1024, 256, 256]" = torch.ops.aten.bmm.default(view_2, arg2_1);  view_2 = arg2_1 = None
        convert_element_type: "f32[1024, 256, 256]" = torch.ops.prims.convert_element_type.default(bmm, torch.float32);  bmm = None
        mul: "f32[1024, 256, 256]" = torch.ops.aten.mul.Tensor(convert_element_type, arg3_1);  convert_element_type = arg3_1 = None
        return mul

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
