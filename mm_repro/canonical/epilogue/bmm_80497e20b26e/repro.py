"""Curated MM repro. Pattern hash: 80497e20b26e."""
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
    def forward(self, arg0_1: "bf16[192, 512, 64]", arg1_1: "bf16[192, 64, 512]", arg2_1: "b8[8, 1, 512, 512]", arg3_1: "bf16[]", _shape_param_0):
        # No stacktrace found for following nodes
        bmm: "bf16[192, 512, 512]" = torch.ops.aten.bmm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        view: "bf16[8, 24, 512, 512]" = torch.ops.aten.view.default(bmm, _shape_param_0);  bmm = _shape_param_0 = None
        where: "bf16[8, 24, 512, 512]" = torch.ops.aten.where.self(arg2_1, arg3_1, view);  arg2_1 = arg3_1 = view = None
        convert_element_type: "f32[8, 24, 512, 512]" = torch.ops.prims.convert_element_type.default(where, torch.float32)
        return (where, convert_element_type)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
