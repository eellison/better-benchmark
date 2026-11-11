"""Curated MM repro. Pattern hash: 0b19e8ffbc5a."""
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
    def forward(self, arg0_1: "bf16[512, 512, 64]", arg1_1: "bf16[512, 64, 512]", arg2_1: "bf16[8, 64, 512, 512]", _shape_param_0):
        # No stacktrace found for following nodes
        bmm: "bf16[512, 512, 512]" = torch.ops.aten.bmm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        view: "bf16[8, 64, 512, 512]" = torch.ops.aten.view.default(bmm, _shape_param_0);  bmm = _shape_param_0 = None
        convert_element_type: "f32[8, 64, 512, 512]" = torch.ops.prims.convert_element_type.default(view, torch.float32);  view = None
        convert_element_type_1: "f32[8, 64, 512, 512]" = torch.ops.prims.convert_element_type.default(arg2_1, torch.float32);  arg2_1 = None
        mul: "f32[8, 64, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type, convert_element_type_1);  convert_element_type = None
        sum_1: "f32[8, 64, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul, [-1], True)
        neg: "f32[8, 64, 512, 512]" = torch.ops.aten.neg.default(convert_element_type_1);  convert_element_type_1 = None
        fma: "f32[8, 64, 512, 512]" = torch.ops.prims.fma.default(neg, sum_1, mul);  neg = sum_1 = mul = None
        convert_element_type_2: "bf16[8, 64, 512, 512]" = torch.ops.prims.convert_element_type.default(fma, torch.bfloat16);  fma = None
        return convert_element_type_2

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
