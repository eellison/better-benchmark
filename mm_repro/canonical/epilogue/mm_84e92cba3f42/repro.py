"""Curated MM repro. Pattern hash: 84e92cba3f42."""
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
    def forward(self, arg0_1: "bf16[128, 1000]", arg1_1: "bf16[1000, 2048]", arg2_1: "b8[128, 2048, 7, 7]", arg3_1: "bf16[128, 2048, 7, 7]", arg4_1: "f32[1, 2048, 1, 1]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        mm: "bf16[128, 2048]" = torch.ops.aten.mm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        view: "bf16[128, 2048, 1, 1]" = torch.ops.aten.view.default(mm, _shape_param_0);  mm = _shape_param_0 = None
        expand: "bf16[128, 2048, 7, 7]" = torch.ops.aten.expand.default(view, _shape_param_1);  view = _shape_param_1 = None
        div: "bf16[128, 2048, 7, 7]" = torch.ops.aten.div.Scalar(expand, 49);  expand = None
        full: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "bf16[128, 2048, 7, 7]" = torch.ops.aten.where.self(arg2_1, full, div);  arg2_1 = full = div = None
        convert_element_type: "f32[128, 2048, 7, 7]" = torch.ops.prims.convert_element_type.default(where, torch.float32)
        convert_element_type_1: "f32[128, 2048, 7, 7]" = torch.ops.prims.convert_element_type.default(arg3_1, torch.float32);  arg3_1 = None
        sub: "f32[128, 2048, 7, 7]" = torch.ops.aten.sub.Tensor(convert_element_type_1, arg4_1);  convert_element_type_1 = arg4_1 = None
        mul: "f32[128, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(convert_element_type, sub);  sub = None
        return (where, convert_element_type, mul)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
