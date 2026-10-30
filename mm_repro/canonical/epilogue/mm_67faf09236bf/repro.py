"""Curated MM repro. Pattern hash: 67faf09236bf."""
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
    def forward(self, arg0_1: "bf16[32768, 128]", arg1_1: "bf16[128, 512]", arg2_1: "b8[256, 128, 512]", arg3_1: "bf16[]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        mm: "bf16[32768, 512]" = torch.ops.aten.mm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        view: "bf16[256, 128, 512]" = torch.ops.aten.view.default(mm, _shape_param_0);  mm = _shape_param_0 = None
        where: "bf16[256, 128, 512]" = torch.ops.aten.where.self(arg2_1, arg3_1, view);  arg2_1 = arg3_1 = view = None
        view_1: "bf16[32768, 512]" = torch.ops.aten.view.default(where, _shape_param_1);  where = _shape_param_1 = None
        permute: "bf16[512, 32768]" = torch.ops.aten.permute.default(view_1, [1, 0])
        return (view_1, permute)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
