"""Curated MM repro. Pattern hash: 4385cc41be3e."""
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
    def forward(self, arg0_1: "bf16[50257, 768]", arg1_1: "bf16[16384, 768]", _shape_param_0):
        # No stacktrace found for following nodes
        permute: "bf16[768, 50257]" = torch.ops.aten.permute.default(arg0_1, [1, 0]);  arg0_1 = None
        constant_pad_nd: "bf16[768, 50264]" = torch.ops.aten.constant_pad_nd.default(permute, _shape_param_0);  permute = _shape_param_0 = None
        mm: "bf16[16384, 50264]" = torch.ops.aten.mm.default(arg1_1, constant_pad_nd);  arg1_1 = constant_pad_nd = None
        return mm

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
