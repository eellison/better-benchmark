"""Curated MM repro. Pattern hash: 3a2ecad2cf2d."""
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
    def forward(self, arg0_1: "f32[1024, 16, 64]", arg1_1: "bf16[8192, 1024]", arg2_1: "f32[1024, 16, 64]", arg3_1: "f32[1024, 16, 64]", arg4_1: "f32[16, 64]", arg5_1: "f32[16, 64]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4):
        # No stacktrace found for following nodes
        convert_element_type: "bf16[1024, 16, 64]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.bfloat16);  arg0_1 = None
        squeeze: "bf16[1024, 1024]" = torch.ops.aten.view.default(convert_element_type, _shape_param_0);  convert_element_type = _shape_param_0 = None
        mm: "bf16[8192, 1024]" = torch.ops.aten.mm.default(arg1_1, squeeze);  squeeze = None
        view_1: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(mm, _shape_param_1);  mm = _shape_param_1 = None
        permute: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_1, [0, 1, 3, 4, 2]);  view_1 = None
        view_2: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute, _shape_param_2);  permute = _shape_param_2 = None
        convert_element_type_1: "bf16[1024, 16, 64]" = torch.ops.prims.convert_element_type.default(arg2_1, torch.bfloat16);  arg2_1 = None
        squeeze_1: "bf16[1024, 1024]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_3);  convert_element_type_1 = _shape_param_3 = None
        mm_1: "bf16[8192, 1024]" = torch.ops.aten.mm.default(arg1_1, squeeze_1);  squeeze_1 = None
        convert_element_type_2: "bf16[1024, 16, 64]" = torch.ops.prims.convert_element_type.default(arg3_1, torch.bfloat16);  arg3_1 = None
        squeeze_2: "bf16[1024, 1024]" = torch.ops.aten.view.default(convert_element_type_2, _shape_param_4);  convert_element_type_2 = _shape_param_4 = None
        mm_2: "bf16[8192, 1024]" = torch.ops.aten.mm.default(arg1_1, squeeze_2);  arg1_1 = squeeze_2 = None
        add: "f32[512, 16, 16, 64]" = torch.ops.aten.add.Tensor(view_2, arg4_1);  arg4_1 = None
        convert_element_type_3: "bf16[512, 16, 16, 64]" = torch.ops.prims.convert_element_type.default(add, torch.bfloat16);  add = None
        add_1: "f32[512, 16, 16, 64]" = torch.ops.aten.add.Tensor(view_2, arg5_1);  view_2 = arg5_1 = None
        convert_element_type_4: "bf16[512, 16, 16, 64]" = torch.ops.prims.convert_element_type.default(add_1, torch.bfloat16);  add_1 = None
        return (mm_1, mm_2, convert_element_type_3, convert_element_type_4)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
