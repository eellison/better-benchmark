"""Curated MM repro. Pattern hash: c48a636cb689."""
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
    def forward(self, arg0_1: "bf16[8192, 512]", arg1_1: "bf16[512, 2048]", arg2_1: "b8[8, 1024, 2048]", arg3_1: "b8[8, 1024, 2048]", arg4_1: "bf16[]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        mm: "bf16[8192, 2048]" = torch.ops.aten.mm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        view: "bf16[8, 1024, 2048]" = torch.ops.aten.view.default(mm, _shape_param_0);  mm = _shape_param_0 = None
        convert_element_type: "bf16[8, 1024, 2048]" = torch.ops.prims.convert_element_type.default(view, torch.bfloat16);  view = None
        convert_element_type_1: "bf16[8, 1024, 2048]" = torch.ops.prims.convert_element_type.default(arg2_1, torch.bfloat16);  arg2_1 = None
        mul: "bf16[8, 1024, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_1, 1.1111111111111112);  convert_element_type_1 = None
        mul_1: "bf16[8, 1024, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type, mul);  convert_element_type = mul = None
        where: "bf16[8, 1024, 2048]" = torch.ops.aten.where.self(arg3_1, arg4_1, mul_1);  arg3_1 = arg4_1 = mul_1 = None
        view_1: "bf16[8192, 2048]" = torch.ops.aten.view.default(where, _shape_param_1);  where = _shape_param_1 = None
        permute: "bf16[2048, 8192]" = torch.ops.aten.permute.default(view_1, [1, 0])
        return (view_1, permute)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
