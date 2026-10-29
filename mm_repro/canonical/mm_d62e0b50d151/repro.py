"""Curated MM repro. Pattern hash: d62e0b50d151."""
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
    def forward(self, arg0_1: "bf16[128, 10]", arg1_1: "bf16[10, 64]", arg2_1: "b8[128, 64, 8, 8]", arg3_1: "bf16[]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        mm: "bf16[128, 64]" = torch.ops.aten.mm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        view: "bf16[128, 64, 1, 1]" = torch.ops.aten.view.default(mm, _shape_param_0);  mm = _shape_param_0 = None
        expand: "bf16[128, 64, 8, 8]" = torch.ops.aten.expand.default(view, _shape_param_1);  view = _shape_param_1 = None
        div: "bf16[128, 64, 8, 8]" = torch.ops.aten.div.Scalar(expand, 64);  expand = None
        where: "bf16[128, 64, 8, 8]" = torch.ops.aten.where.self(arg2_1, arg3_1, div);  arg2_1 = arg3_1 = div = None
        convert_element_type: "f32[128, 64, 8, 8]" = torch.ops.prims.convert_element_type.default(where, torch.float32)
        return (where, convert_element_type)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
