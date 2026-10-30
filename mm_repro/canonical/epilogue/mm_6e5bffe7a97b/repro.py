"""Curated MM repro. Pattern hash: 6e5bffe7a97b."""
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
    def forward(self, arg0_1: "bf16[128, 1000]", arg1_1: "bf16[1000, 512]", arg2_1: "b8[128, 512, 1, 1]", arg3_1: "bf16[128, 512, 1, 1]", arg4_1: "f32[128, 32]", arg5_1: "f32[512]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6):
        # No stacktrace found for following nodes
        mm: "bf16[128, 512]" = torch.ops.aten.mm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        convert_element_type: "f32[128, 512]" = torch.ops.prims.convert_element_type.default(mm, torch.float32);  mm = None
        view: "f32[128, 512, 1, 1]" = torch.ops.aten.view.default(convert_element_type, _shape_param_0);  convert_element_type = _shape_param_0 = None
        div: "f32[128, 512, 1, 1]" = torch.ops.aten.div.Scalar(view, 1);  view = None
        full: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "f32[128, 512, 1, 1]" = torch.ops.aten.where.self(arg2_1, full, div);  arg2_1 = full = div = None
        convert_element_type_1: "f32[128, 512, 1, 1]" = torch.ops.prims.convert_element_type.default(arg3_1, torch.float32);  arg3_1 = None
        mul: "f32[128, 512, 1, 1]" = torch.ops.aten.mul.Tensor(where, convert_element_type_1);  convert_element_type_1 = None
        view_1: "f32[128, 512, 1]" = torch.ops.aten.view.default(mul, _shape_param_1);  mul = _shape_param_1 = None
        view_2: "f32[128, 512, 1]" = torch.ops.aten.view.default(where, _shape_param_2);  _shape_param_2 = None
        unsqueeze: "f32[128, 32, 1]" = torch.ops.aten.view.default(arg4_1, _shape_param_3);  arg4_1 = _shape_param_3 = None
        view_3: "f32[1, 32, 16]" = torch.ops.aten.view.default(arg5_1, _shape_param_4);  arg5_1 = _shape_param_4 = None
        mul_1: "f32[128, 32, 16]" = torch.ops.aten.mul.Tensor(unsqueeze, view_3);  unsqueeze = view_3 = None
        unsqueeze_1: "f32[128, 32, 16, 1]" = torch.ops.aten.view.default(mul_1, _shape_param_5);  mul_1 = _shape_param_5 = None
        view_4: "f32[128, 32, 16, 1]" = torch.ops.aten.view.default(where, _shape_param_6);  _shape_param_6 = None
        mul_2: "f32[128, 32, 16, 1]" = torch.ops.aten.mul.Tensor(view_4, unsqueeze_1);  view_4 = unsqueeze_1 = None
        return (where, view_1, view_2, mul_2)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
