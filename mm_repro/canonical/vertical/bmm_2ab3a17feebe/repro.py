"""Curated MM repro. Pattern hash: 2ab3a17feebe."""
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
    def forward(self, arg0_1: "f32[128, 6, 49, 49]", arg1_1: "f32[128, 6, 49, 49]", arg2_1: "bf16[768, 128, 49]", arg3_1: "bf16[768, 49, 128]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        sum_1: "f32[128, 6, 49, 1]" = torch.ops.aten.sum.dim_IntList(arg0_1, [3], True)
        neg: "f32[128, 6, 49, 49]" = torch.ops.aten.neg.default(arg1_1);  arg1_1 = None
        fma: "f32[128, 6, 49, 49]" = torch.ops.prims.fma.default(neg, sum_1, arg0_1);  neg = sum_1 = arg0_1 = None
        convert_element_type: "bf16[128, 6, 49, 49]" = torch.ops.prims.convert_element_type.default(fma, torch.bfloat16);  fma = None
        mul: "bf16[128, 6, 49, 49]" = torch.ops.aten.mul.Tensor(convert_element_type, 0.08838834764831845);  convert_element_type = None
        view: "bf16[768, 49, 49]" = torch.ops.aten.view.default(mul, _shape_param_0);  mul = _shape_param_0 = None
        bmm: "bf16[768, 128, 49]" = torch.ops.aten.bmm.default(arg2_1, view);  arg2_1 = None
        bmm_1: "bf16[768, 49, 128]" = torch.ops.aten.bmm.default(view, arg3_1);  view = arg3_1 = None
        view_1: "bf16[128, 6, 128, 49]" = torch.ops.aten.view.default(bmm, _shape_param_1);  bmm = _shape_param_1 = None
        view_2: "bf16[128, 6, 49, 128]" = torch.ops.aten.view.default(bmm_1, _shape_param_2);  bmm_1 = _shape_param_2 = None
        permute: "bf16[128, 6, 49, 128]" = torch.ops.aten.permute.default(view_1, [0, 1, 3, 2]);  view_1 = None
        return (view_2, permute)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
