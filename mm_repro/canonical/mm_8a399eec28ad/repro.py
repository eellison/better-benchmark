"""Curated MM repro. Pattern hash: 8a399eec28ad."""
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
    def forward(self, arg0_1: "f32[64, 128, 1024]", arg1_1: "b8[64, 128, 1024]", arg2_1: "bf16[1024, 4096]", arg3_1: "b8[64, 128, 4096]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        convert_element_type: "bf16[64, 128, 1024]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.bfloat16);  arg0_1 = None
        convert_element_type_1: "bf16[64, 128, 1024]" = torch.ops.prims.convert_element_type.default(arg1_1, torch.bfloat16);  arg1_1 = None
        mul: "bf16[64, 128, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_1, 1.1111111111111112);  convert_element_type_1 = None
        mul_1: "bf16[64, 128, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type, mul);  convert_element_type = mul = None
        view: "bf16[8192, 1024]" = torch.ops.aten.view.default(mul_1, _shape_param_0);  mul_1 = _shape_param_0 = None
        mm: "bf16[8192, 4096]" = torch.ops.aten.mm.default(view, arg2_1);  view = arg2_1 = None
        view_1: "bf16[64, 128, 4096]" = torch.ops.aten.view.default(mm, _shape_param_1);  mm = _shape_param_1 = None
        full: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "bf16[64, 128, 4096]" = torch.ops.aten.where.self(arg3_1, full, view_1);  arg3_1 = full = view_1 = None
        view_2: "bf16[8192, 4096]" = torch.ops.aten.view.default(where, _shape_param_2);  where = _shape_param_2 = None
        permute: "bf16[4096, 8192]" = torch.ops.aten.permute.default(view_2, [1, 0])
        return (view_2, permute)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
