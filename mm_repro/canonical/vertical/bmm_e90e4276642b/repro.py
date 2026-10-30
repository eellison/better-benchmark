"""Curated MM repro. Pattern hash: e90e4276642b."""
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
    def forward(self, arg0_1: "f32[128, 32, 49, 49]", arg1_1: "f32[128, 32, 49, 49]", arg2_1: "bf16[4096, 32, 49]", arg3_1: "bf16[4096, 49, 32]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        sum_1: "f32[128, 32, 49, 1]" = torch.ops.aten.sum.dim_IntList(arg0_1, [3], True)
        neg: "f32[128, 32, 49, 49]" = torch.ops.aten.neg.default(arg1_1);  arg1_1 = None
        fma: "f32[128, 32, 49, 49]" = torch.ops.prims.fma.default(neg, sum_1, arg0_1);  neg = sum_1 = arg0_1 = None
        convert_element_type: "bf16[128, 32, 49, 49]" = torch.ops.prims.convert_element_type.default(fma, torch.bfloat16)
        view: "bf16[4096, 49, 49]" = torch.ops.aten.view.default(convert_element_type, _shape_param_0);  convert_element_type = _shape_param_0 = None
        bmm: "bf16[4096, 32, 49]" = torch.ops.aten.bmm.default(arg2_1, view);  arg2_1 = None
        bmm_1: "bf16[4096, 49, 32]" = torch.ops.aten.bmm.default(view, arg3_1);  view = arg3_1 = None
        view_1: "bf16[128, 32, 32, 49]" = torch.ops.aten.view.default(bmm, _shape_param_1);  bmm = _shape_param_1 = None
        view_2: "bf16[128, 32, 49, 32]" = torch.ops.aten.view.default(bmm_1, _shape_param_2);  bmm_1 = _shape_param_2 = None
        permute: "bf16[128, 32, 49, 32]" = torch.ops.aten.permute.default(view_1, [0, 1, 3, 2]);  view_1 = None
        mul: "bf16[128, 32, 49, 32]" = torch.ops.aten.mul.Tensor(view_2, 0.1767766952966369);  view_2 = None
        return (fma, permute, mul)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
