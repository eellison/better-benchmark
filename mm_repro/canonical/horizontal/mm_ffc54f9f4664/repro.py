"""Curated MM repro. Pattern hash: ffc54f9f4664."""
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
    def forward(self, arg0_1: "f32[1024, 16, 64]", arg1_1: "bf16[8192, 1024]", arg2_1: "f32[1024, 16, 64]", arg3_1: "f32[1024, 16, 64]", arg4_1: "f32[16, 64]", arg5_1: "f32[16, 64]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13, _shape_param_14):
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
        view_4: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(mm_1, _shape_param_4);  mm_1 = _shape_param_4 = None
        permute_1: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_4, [0, 1, 3, 4, 2]);  view_4 = None
        view_5: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_1, _shape_param_5);  permute_1 = _shape_param_5 = None
        convert_element_type_2: "bf16[1024, 16, 64]" = torch.ops.prims.convert_element_type.default(arg3_1, torch.bfloat16);  arg3_1 = None
        squeeze_2: "bf16[1024, 1024]" = torch.ops.aten.view.default(convert_element_type_2, _shape_param_6);  convert_element_type_2 = _shape_param_6 = None
        mm_2: "bf16[8192, 1024]" = torch.ops.aten.mm.default(arg1_1, squeeze_2);  arg1_1 = squeeze_2 = None
        view_7: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(mm_2, _shape_param_7);  mm_2 = _shape_param_7 = None
        permute_2: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_7, [0, 1, 3, 4, 2]);  view_7 = None
        view_8: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_2, _shape_param_8);  permute_2 = _shape_param_8 = None
        add: "f32[512, 16, 16, 64]" = torch.ops.aten.add.Tensor(view_2, arg4_1);  arg4_1 = None
        convert_element_type_3: "bf16[512, 16, 16, 64]" = torch.ops.prims.convert_element_type.default(add, torch.bfloat16);  add = None
        unsqueeze_9: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.view.default(convert_element_type_3, _shape_param_9);  convert_element_type_3 = _shape_param_9 = None
        permute_3: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_9, [1, 2, 0, 4, 3]);  unsqueeze_9 = None
        unsqueeze_10: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.view.default(view_5, _shape_param_10);  view_5 = _shape_param_10 = None
        permute_4: "bf16[16, 16, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_10, [1, 2, 4, 0, 3]);  unsqueeze_10 = None
        permute_5: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_3, [0, 1, 2, 4, 3]);  permute_3 = None
        permute_6: "bf16[16, 16, 64, 512, 1]" = torch.ops.aten.permute.default(permute_4, [0, 1, 4, 3, 2]);  permute_4 = None
        view_9: "bf16[256, 64, 512]" = torch.ops.aten.view.default(permute_6, _shape_param_11);  permute_6 = _shape_param_11 = None
        add_1: "f32[512, 16, 16, 64]" = torch.ops.aten.add.Tensor(view_2, arg5_1);  view_2 = arg5_1 = None
        convert_element_type_4: "bf16[512, 16, 16, 64]" = torch.ops.prims.convert_element_type.default(add_1, torch.bfloat16);  add_1 = None
        unsqueeze_11: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.view.default(convert_element_type_4, _shape_param_12);  convert_element_type_4 = _shape_param_12 = None
        permute_7: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_11, [1, 2, 0, 4, 3]);  unsqueeze_11 = None
        permute_8: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_7, [0, 1, 2, 4, 3]);  permute_7 = None
        unsqueeze_12: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.view.default(view_8, _shape_param_13);  view_8 = _shape_param_13 = None
        permute_9: "bf16[1, 16, 16, 64, 512]" = torch.ops.aten.permute.default(unsqueeze_12, [4, 1, 2, 3, 0]);  unsqueeze_12 = None
        permute_10: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_9, [1, 2, 4, 3, 0]);  permute_9 = None
        view_10: "bf16[256, 512, 64]" = torch.ops.aten.view.default(permute_10, _shape_param_14);  permute_10 = _shape_param_14 = None
        return (permute_5, view_9, permute_8, view_10)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
