"""Curated MM repro. Pattern hash: 0d0980e3f195."""
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
    def forward(self, arg0_1: "bf16[128, 1000]", arg1_1: "bf16[1000, 1024]", arg2_1: "b8[128, 1024, 7, 7]", arg3_1: "bf16[]", arg4_1: "f32[128, 1024, 7, 7]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        mm: "bf16[128, 1024]" = torch.ops.aten.mm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        unsqueeze_1: "bf16[128, 1024, 1, 1]" = torch.ops.aten.view.default(mm, _shape_param_0);  mm = _shape_param_0 = None
        expand: "bf16[128, 1024, 7, 7]" = torch.ops.aten.expand.default(unsqueeze_1, _shape_param_1);  unsqueeze_1 = _shape_param_1 = None
        div: "bf16[128, 1024, 7, 7]" = torch.ops.aten.div.Scalar(expand, 49);  expand = None
        where: "bf16[128, 1024, 7, 7]" = torch.ops.aten.where.self(arg2_1, arg3_1, div);  arg2_1 = arg3_1 = div = None
        convert_element_type: "f32[128, 1024, 7, 7]" = torch.ops.prims.convert_element_type.default(where, torch.float32);  where = None
        mul: "f32[128, 1024, 7, 7]" = torch.ops.aten.mul.Tensor(convert_element_type, arg4_1);  arg4_1 = None
        return (convert_element_type, mul)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
