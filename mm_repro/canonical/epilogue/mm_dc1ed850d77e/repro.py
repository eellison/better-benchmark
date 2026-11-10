"""Curated MM repro. Pattern hash: dc1ed850d77e."""
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
    def forward(self, arg0_1: "bf16[128, 4096]", arg1_1: "bf16[4096, 4096]", arg2_1: "b8[128, 4096]", arg3_1: "b8[128, 4096]", arg4_1: "bf16[]", _shape_param_0):
        # No stacktrace found for following nodes
        mm: "bf16[128, 4096]" = torch.ops.aten.mm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        convert_element_type: "bf16[128, 4096]" = torch.ops.prims.convert_element_type.default(arg2_1, torch.bfloat16);  arg2_1 = None
        mul: "bf16[128, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type, 2.0);  convert_element_type = None
        mul_1: "bf16[128, 4096]" = torch.ops.aten.mul.Tensor(mm, mul);  mm = mul = None
        where: "bf16[128, 4096]" = torch.ops.aten.where.self(arg3_1, arg4_1, mul_1);  arg3_1 = arg4_1 = mul_1 = None
        permute: "bf16[4096, 128]" = torch.ops.aten.permute.default(where, [1, 0])
        sum_1: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(where, [0], True, dtype = torch.float32)
        view: "f32[4096]" = torch.ops.aten.view.default(sum_1, _shape_param_0);  sum_1 = _shape_param_0 = None
        convert_element_type_1: "bf16[4096]" = torch.ops.prims.convert_element_type.default(view, torch.bfloat16);  view = None
        convert_element_type_2: "f32[4096]" = torch.ops.prims.convert_element_type.default(convert_element_type_1, torch.float32);  convert_element_type_1 = None
        return (where, permute, convert_element_type_2)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
