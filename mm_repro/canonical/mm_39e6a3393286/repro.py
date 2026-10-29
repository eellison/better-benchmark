"""Curated MM repro. Pattern hash: 39e6a3393286."""
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
    def forward(self, arg0_1: "bf16[2560, 2048]", arg1_1: "bf16[2048, 2560]", arg2_1: "bf16[2560, 2048]"):
        # No stacktrace found for following nodes
        mm: "bf16[2560, 2560]" = torch.ops.aten.mm.default(arg0_1, arg1_1);  arg0_1 = None
        convert_element_type: "f32[2560, 2560]" = torch.ops.prims.convert_element_type.default(mm, torch.float32);  mm = None
        mm_1: "bf16[2560, 2560]" = torch.ops.aten.mm.default(arg2_1, arg1_1);  arg2_1 = arg1_1 = None
        convert_element_type_1: "f32[2560, 2560]" = torch.ops.prims.convert_element_type.default(mm_1, torch.float32);  mm_1 = None
        return (convert_element_type, convert_element_type_1)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
