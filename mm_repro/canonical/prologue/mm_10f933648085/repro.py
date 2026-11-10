"""Curated MM repro. Pattern hash: 10f933648085."""
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
    def forward(self, arg0_1: "f32[50257, 768]", arg1_1: "bf16[16384, 768]", _shape_param_0):
        # No stacktrace found for following nodes
        convert_element_type: "bf16[50257, 768]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.bfloat16);  arg0_1 = None
        permute: "bf16[768, 50257]" = torch.ops.aten.permute.default(convert_element_type, [1, 0]);  convert_element_type = None
        constant_pad_nd: "bf16[768, 50264]" = torch.ops.aten.constant_pad_nd.default(permute, _shape_param_0);  _shape_param_0 = None
        mm: "bf16[16384, 50264]" = torch.ops.aten.mm.default(arg1_1, constant_pad_nd);  arg1_1 = constant_pad_nd = None
        permute_1: "bf16[50257, 768]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
        return (mm, permute_1)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
