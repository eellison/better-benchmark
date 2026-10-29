"""Curated MM repro. Pattern hash: 928492c458b4."""
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
    def forward(self, arg0_1: "bf16[512, 1000]", arg1_1: "bf16[1000, 1280]", arg2_1: "f32[512, 1280, 1, 1]", arg3_1: "b8[512, 1280, 1, 1]", arg4_1: "b8[512, 1280, 1, 1]", _shape_param_0):
        # No stacktrace found for following nodes
        mm: "bf16[512, 1280]" = torch.ops.aten.mm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        view: "bf16[512, 1280, 1, 1]" = torch.ops.aten.view.default(mm, _shape_param_0);  mm = _shape_param_0 = None
        convert_element_type: "f32[512, 1280, 1, 1]" = torch.ops.prims.convert_element_type.default(view, torch.float32);  view = None
        mul: "f32[512, 1280, 1, 1]" = torch.ops.aten.mul.Tensor(convert_element_type, arg2_1);  arg2_1 = None
        where: "f32[512, 1280, 1, 1]" = torch.ops.aten.where.self(arg3_1, mul, convert_element_type);  arg3_1 = mul = convert_element_type = None
        full: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1: "f32[512, 1280, 1, 1]" = torch.ops.aten.where.self(arg4_1, full, where);  arg4_1 = full = where = None
        convert_element_type_1: "bf16[512, 1280, 1, 1]" = torch.ops.prims.convert_element_type.default(where_1, torch.bfloat16);  where_1 = None
        return convert_element_type_1

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
