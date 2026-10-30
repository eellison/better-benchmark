"""Curated MM repro. Pattern hash: 915caaf3b91e."""
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
    def forward(self, arg0_1: "f32[768, 2]", arg1_1: "bf16[8192, 768]", arg2_1: "i64[8]", arg3_1: "i64[8]", _shape_param_0):
        # No stacktrace found for following nodes
        convert_element_type: "bf16[768, 2]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.bfloat16);  arg0_1 = None
        mm: "bf16[8192, 2]" = torch.ops.aten.mm.default(arg1_1, convert_element_type);  arg1_1 = convert_element_type = None
        view: "bf16[8, 1024, 2]" = torch.ops.aten.view.default(mm, _shape_param_0);  mm = _shape_param_0 = None
        index: "bf16[8, 2]" = torch.ops.aten.index.Tensor(view, [arg2_1, arg3_1]);  view = arg2_1 = arg3_1 = None
        convert_element_type_1: "f32[8, 2]" = torch.ops.prims.convert_element_type.default(index, torch.float32)
        return (index, convert_element_type_1)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
