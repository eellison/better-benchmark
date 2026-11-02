"""Curated MM repro. Pattern hash: 6658ff8c5987."""
import sys
from pathlib import Path
from math import inf, nan
import torch
import torch._inductor.inductor_prims
from torch import device
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from repro_harness import benchmark_repro, load_shape_configs, make_inputs_from_config
from mm_replay import make_inputs_from_config, benchmark_repro_safe as benchmark_repro
_repro_version = 3

class Repro(torch.nn.Module):
    def forward(self, arg0_1: "f32[4096, 128]", arg1_1: "bf16[4096]", arg2_1: "bf16[4096, 128]"):
        # No stacktrace found for following nodes
        convert_element_type: "bf16[4096, 128]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.bfloat16);  arg0_1 = None
        permute: "bf16[128, 4096]" = torch.ops.aten.permute.default(convert_element_type, [1, 0]);  convert_element_type = None
        addmm: "bf16[4096, 4096]" = torch.ops.aten.addmm.default(arg1_1, arg2_1, permute);  arg1_1 = arg2_1 = None
        permute_1: "bf16[4096, 128]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
        return (addmm, permute_1)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
