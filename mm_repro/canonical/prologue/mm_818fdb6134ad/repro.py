"""Curated MM repro. Pattern hash: 818fdb6134ad."""
import sys
from pathlib import Path
from math import inf, nan
import torch
import torch._inductor.inductor_prims
from torch import device
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from repro_harness import benchmark_repro, load_shape_configs, make_inputs_from_config
from mm_replay import make_inputs_from_config, benchmark_repro_safe as benchmark_repro
_repro_version = 3

class Repro(torch.nn.Module):
    def forward(self, arg0_1: "bf16[8192, 2048]", arg1_1: "b8[8, 1024, 2048]", arg2_1: "b8[8, 1024, 2048]", arg3_1: "bf16[]", arg4_1: "bf16[8192, 512]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        view: "bf16[8, 1024, 2048]" = torch.ops.aten.view.default(arg0_1, _shape_param_0);  arg0_1 = _shape_param_0 = None
        convert_element_type: "bf16[8, 1024, 2048]" = torch.ops.prims.convert_element_type.default(view, torch.bfloat16);  view = None
        convert_element_type_1: "bf16[8, 1024, 2048]" = torch.ops.prims.convert_element_type.default(arg1_1, torch.bfloat16);  arg1_1 = None
        mul: "bf16[8, 1024, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_1, 1.1111111111111112);  convert_element_type_1 = None
        mul_1: "bf16[8, 1024, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type, mul);  convert_element_type = mul = None
        where: "bf16[8, 1024, 2048]" = torch.ops.aten.where.self(arg2_1, arg3_1, mul_1);  arg2_1 = arg3_1 = mul_1 = None
        view_1: "bf16[8192, 2048]" = torch.ops.aten.view.default(where, _shape_param_1);  where = _shape_param_1 = None
        permute: "bf16[2048, 8192]" = torch.ops.aten.permute.default(view_1, [1, 0])
        mm: "bf16[2048, 512]" = torch.ops.aten.mm.default(permute, arg4_1);  permute = arg4_1 = None
        return (view_1, mm)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
