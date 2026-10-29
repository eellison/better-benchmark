"""Curated MM repro. Pattern hash: 7118a59919a7."""
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
    def forward(self, arg0_1: "bf16[8192, 768]", arg1_1: "bf16[768, 768]", arg2_1: "bf16[6291456]", arg3_1: "bf16[768, 768]", arg4_1: "bf16[8192, 768]", arg5_1: "bf16[768, 768]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7):
        # No stacktrace found for following nodes
        mm: "bf16[8192, 768]" = torch.ops.aten.mm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        view: "bf16[1024, 8, 768]" = torch.ops.aten.view.default(mm, _shape_param_0);  mm = _shape_param_0 = None
        convert_element_type: "f32[1024, 8, 768]" = torch.ops.prims.convert_element_type.default(view, torch.float32);  view = None
        as_strided: "bf16[96, 2, 512, 64]" = torch.ops.aten.as_strided.default(arg2_1, _shape_param_1, _shape_param_2, 0);  arg2_1 = _shape_param_1 = _shape_param_2 = None
        view_1: "bf16[96, 1024, 64]" = torch.ops.aten.view.default(as_strided, _shape_param_3);  as_strided = _shape_param_3 = None
        view_2: "bf16[8, 12, 1024, 64]" = torch.ops.aten.view.default(view_1, _shape_param_4);  view_1 = _shape_param_4 = None
        permute: "bf16[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_2, [0, 2, 1, 3]);  view_2 = None
        permute_1: "bf16[1024, 8, 12, 64]" = torch.ops.aten.permute.default(permute, [1, 0, 2, 3]);  permute = None
        view_4: "bf16[8192, 768]" = torch.ops.aten.view.default(permute_1, _shape_param_5);  permute_1 = _shape_param_5 = None
        mm_1: "bf16[8192, 768]" = torch.ops.aten.mm.default(view_4, arg3_1);  view_4 = arg3_1 = None
        view_5: "bf16[1024, 8, 768]" = torch.ops.aten.view.default(mm_1, _shape_param_6);  mm_1 = _shape_param_6 = None
        convert_element_type_1: "f32[1024, 8, 768]" = torch.ops.prims.convert_element_type.default(view_5, torch.float32);  view_5 = None
        add: "f32[1024, 8, 768]" = torch.ops.aten.add.Tensor(convert_element_type, convert_element_type_1);  convert_element_type = convert_element_type_1 = None
        mm_2: "bf16[8192, 768]" = torch.ops.aten.mm.default(arg4_1, arg5_1);  arg4_1 = arg5_1 = None
        view_6: "bf16[1024, 8, 768]" = torch.ops.aten.view.default(mm_2, _shape_param_7);  mm_2 = _shape_param_7 = None
        convert_element_type_2: "f32[1024, 8, 768]" = torch.ops.prims.convert_element_type.default(view_6, torch.float32);  view_6 = None
        add_1: "f32[1024, 8, 768]" = torch.ops.aten.add.Tensor(add, convert_element_type_2);  add = convert_element_type_2 = None
        permute_2: "f32[8, 1024, 768]" = torch.ops.aten.permute.default(add_1, [1, 0, 2]);  add_1 = None
        return permute_2

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
