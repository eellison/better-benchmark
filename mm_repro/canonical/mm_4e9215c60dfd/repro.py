"""Curated MM repro. Pattern hash: 4e9215c60dfd."""
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
    def forward(self, arg0_1: "bf16[128, 1000]", arg1_1: "bf16[1000, 768]", arg2_1: "f32[98304]", arg3_1: "f32[128, 768, 7, 7]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4):
        # No stacktrace found for following nodes
        mm: "bf16[128, 768]" = torch.ops.aten.mm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        convert_element_type: "f32[128, 768]" = torch.ops.prims.convert_element_type.default(mm, torch.float32);  mm = None
        as_strided_scatter: "f32[98304]" = torch.ops.aten.as_strided_scatter.default(arg2_1, convert_element_type, _shape_param_0, _shape_param_1, 0);  arg2_1 = convert_element_type = _shape_param_0 = _shape_param_1 = None
        as_strided: "f32[128, 768, 1, 1]" = torch.ops.aten.as_strided.default(as_strided_scatter, _shape_param_2, _shape_param_3, 0);  as_strided_scatter = _shape_param_2 = _shape_param_3 = None
        expand: "f32[128, 768, 7, 7]" = torch.ops.aten.expand.default(as_strided, _shape_param_4);  as_strided = _shape_param_4 = None
        div: "f32[128, 768, 7, 7]" = torch.ops.aten.div.Scalar(expand, 49);  expand = None
        mul: "f32[128, 768, 7, 7]" = torch.ops.aten.mul.Tensor(div, arg3_1);  arg3_1 = None
        return (div, mul)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
