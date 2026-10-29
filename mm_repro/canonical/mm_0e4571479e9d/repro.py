"""Curated MM repro. Pattern hash: 0e4571479e9d."""
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
    def forward(self, arg0_1: "bf16[32, 1280]", arg1_1: "bf16[1280, 960]", arg2_1: "f32[32, 960, 7, 7]", arg3_1: "b8[32, 960, 7, 7]", arg4_1: "b8[32, 960, 7, 7]", arg5_1: "f32[]", arg6_1: "f32[32, 960, 7, 7]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        mm: "bf16[32, 960]" = torch.ops.aten.mm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        view: "bf16[32, 960, 1, 1]" = torch.ops.aten.view.default(mm, _shape_param_0);  mm = _shape_param_0 = None
        expand: "bf16[32, 960, 7, 7]" = torch.ops.aten.expand.default(view, _shape_param_1);  view = _shape_param_1 = None
        div: "bf16[32, 960, 7, 7]" = torch.ops.aten.div.Scalar(expand, 49);  expand = None
        convert_element_type: "f32[32, 960, 7, 7]" = torch.ops.prims.convert_element_type.default(div, torch.float32);  div = None
        mul: "f32[32, 960, 7, 7]" = torch.ops.aten.mul.Tensor(convert_element_type, arg2_1);  arg2_1 = None
        where: "f32[32, 960, 7, 7]" = torch.ops.aten.where.self(arg3_1, mul, convert_element_type);  arg3_1 = mul = convert_element_type = None
        where_1: "f32[32, 960, 7, 7]" = torch.ops.aten.where.self(arg4_1, arg5_1, where);  arg4_1 = arg5_1 = where = None
        convert_element_type_1: "bf16[32, 960, 7, 7]" = torch.ops.prims.convert_element_type.default(where_1, torch.bfloat16);  where_1 = None
        convert_element_type_2: "f32[32, 960, 7, 7]" = torch.ops.prims.convert_element_type.default(convert_element_type_1, torch.float32);  convert_element_type_1 = None
        mul_1: "f32[32, 960, 7, 7]" = torch.ops.aten.mul.Tensor(convert_element_type_2, arg6_1);  arg6_1 = None
        return (convert_element_type_2, mul_1)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
