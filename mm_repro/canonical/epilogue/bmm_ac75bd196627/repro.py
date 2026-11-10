"""Curated MM repro. Pattern hash: ac75bd196627."""
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
    def forward(self, arg0_1: "bf16[16, 128, 256]", arg1_1: "bf16[16, 256, 128]", arg2_1: "f32[1, 16, 128, 128]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        bmm: "bf16[16, 128, 128]" = torch.ops.aten.bmm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        view: "bf16[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm, _shape_param_0);  bmm = _shape_param_0 = None
        convert_element_type: "f32[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view, torch.float32);  view = None
        mul: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type, arg2_1);  convert_element_type = None
        sum_1: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul, [-1], True)
        neg: "f32[1, 16, 128, 128]" = torch.ops.aten.neg.default(arg2_1);  arg2_1 = None
        fma: "f32[1, 16, 128, 128]" = torch.ops.prims.fma.default(neg, sum_1, mul);  neg = sum_1 = mul = None
        convert_element_type_1: "bf16[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(fma, torch.bfloat16);  fma = None
        div: "bf16[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(convert_element_type_1, 16.0);  convert_element_type_1 = None
        view_1: "bf16[16, 128, 128]" = torch.ops.aten.view.default(div, _shape_param_1);  div = _shape_param_1 = None
        return view_1

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
