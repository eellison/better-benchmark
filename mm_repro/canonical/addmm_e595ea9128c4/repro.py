"""Curated MM repro. Pattern hash: e595ea9128c4."""
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
    def forward(self, arg0_1: "bf16[128, 2304, 9, 9]", arg1_1: "bf16[2304, 1000]", arg2_1: "bf16[1000]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        convert_element_type: "f32[128, 2304, 9, 9]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.float32);  arg0_1 = None
        neg: "f32[128, 2304, 9, 9]" = torch.ops.aten.neg.default(convert_element_type)
        exp: "f32[128, 2304, 9, 9]" = torch.ops.aten.exp.default(neg);  neg = None
        add: "f32[128, 2304, 9, 9]" = torch.ops.aten.add.Tensor(exp, 1);  exp = None
        div: "f32[128, 2304, 9, 9]" = torch.ops.aten.div.Tensor(convert_element_type, add);  convert_element_type = add = None
        convert_element_type_1: "bf16[128, 2304, 9, 9]" = torch.ops.prims.convert_element_type.default(div, torch.bfloat16);  div = None
        mean: "bf16[128, 2304, 1, 1]" = torch.ops.aten.mean.dim(convert_element_type_1, [2, 3], True);  convert_element_type_1 = None
        as_strided: "bf16[128, 2304, 1, 1]" = torch.ops.aten.as_strided.default(mean, _shape_param_0, _shape_param_1);  mean = _shape_param_0 = _shape_param_1 = None
        view: "bf16[128, 2304]" = torch.ops.aten.view.default(as_strided, _shape_param_2);  as_strided = _shape_param_2 = None
        addmm: "bf16[128, 1000]" = torch.ops.aten.addmm.default(arg2_1, view, arg1_1);  arg2_1 = view = arg1_1 = None
        return addmm

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
