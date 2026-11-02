"""Curated MM repro. Pattern hash: e3f75adba3ac."""
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
    def forward(self, arg0_1: "f32[768, 2304]", arg1_1: "bf16[2304]", arg2_1: "bf16[16384, 768]"):
        # No stacktrace found for following nodes
        convert_element_type: "bf16[768, 2304]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.bfloat16);  arg0_1 = None
        addmm: "bf16[16384, 2304]" = torch.ops.aten.addmm.default(arg1_1, arg2_1, convert_element_type);  arg1_1 = arg2_1 = None
        permute: "bf16[2304, 768]" = torch.ops.aten.permute.default(convert_element_type, [1, 0]);  convert_element_type = None
        return (addmm, permute)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
