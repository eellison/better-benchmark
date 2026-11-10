"""Curated MM repro. Pattern hash: f566fdebbfd4."""
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
    def forward(self, arg0_1: "bf16[50272, 8192]", arg1_1: "bf16[8192, 768]"):
        # No stacktrace found for following nodes
        mm: "bf16[50272, 768]" = torch.ops.aten.mm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        slice_1: "bf16[50265, 768]" = torch.ops.aten.slice.Tensor(mm, 0, 0, -7);  mm = None
        convert_element_type: "f32[50265, 768]" = torch.ops.prims.convert_element_type.default(slice_1, torch.float32);  slice_1 = None
        return convert_element_type

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
