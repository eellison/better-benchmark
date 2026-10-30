"""Curated MM repro. Pattern hash: 61e944c551a1."""
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
    def forward(self, arg0_1: "bf16[128, 1000]", arg1_1: "bf16[1000, 1024]", arg2_1: "f32[1024]", arg3_1: "bf16[128, 7, 7, 1024]", arg4_1: "f32[128, 7, 7, 1]", arg5_1: "f32[128, 7, 7, 1]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        mm: "bf16[128, 1024]" = torch.ops.aten.mm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        convert_element_type: "f32[128, 1024]" = torch.ops.prims.convert_element_type.default(mm, torch.float32);  mm = None
        unsqueeze_1: "f32[128, 1, 1, 1024]" = torch.ops.aten.view.default(convert_element_type, _shape_param_0);  convert_element_type = _shape_param_0 = None
        expand: "f32[128, 7, 7, 1024]" = torch.ops.aten.expand.default(unsqueeze_1, _shape_param_1);  unsqueeze_1 = _shape_param_1 = None
        div: "f32[128, 7, 7, 1024]" = torch.ops.aten.div.Scalar(expand, 49);  expand = None
        mul: "f32[128, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(div, arg2_1);  arg2_1 = None
        mul_1: "f32[128, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(mul, 1024)
        convert_element_type_1: "f32[128, 7, 7, 1024]" = torch.ops.prims.convert_element_type.default(arg3_1, torch.float32);  arg3_1 = None
        sub: "f32[128, 7, 7, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_1, arg4_1);  convert_element_type_1 = arg4_1 = None
        mul_2: "f32[128, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(sub, arg5_1);  sub = arg5_1 = None
        mul_3: "f32[128, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(div, mul_2);  mul_2 = None
        return (div, mul, mul_1, mul_3)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
