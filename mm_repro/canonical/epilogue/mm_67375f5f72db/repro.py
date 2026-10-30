"""Curated MM repro. Pattern hash: 67375f5f72db."""
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
    def forward(self, arg0_1: "f32[1024, 16, 64]", arg1_1: "bf16[8192, 1024]", arg2_1: "f32[16, 64]", arg3_1: "f32[16, 64]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6):
        # No stacktrace found for following nodes
        convert_element_type: "bf16[1024, 16, 64]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.bfloat16);  arg0_1 = None
        squeeze: "bf16[1024, 1024]" = torch.ops.aten.view.default(convert_element_type, _shape_param_0);  convert_element_type = _shape_param_0 = None
        mm: "bf16[8192, 1024]" = torch.ops.aten.mm.default(arg1_1, squeeze);  arg1_1 = squeeze = None
        view_1: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(mm, _shape_param_1);  mm = _shape_param_1 = None
        permute: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_1, [0, 1, 3, 4, 2]);  view_1 = None
        view_2: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute, _shape_param_2);  permute = _shape_param_2 = None
        add: "f32[512, 16, 16, 64]" = torch.ops.aten.add.Tensor(view_2, arg2_1);  arg2_1 = None
        convert_element_type_1: "bf16[512, 16, 16, 64]" = torch.ops.prims.convert_element_type.default(add, torch.bfloat16);  add = None
        unsqueeze_3: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_3);  convert_element_type_1 = _shape_param_3 = None
        permute_1: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_3, [1, 2, 0, 4, 3]);  unsqueeze_3 = None
        permute_2: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_1, [0, 1, 2, 4, 3]);  permute_1 = None
        view_3: "bf16[256, 512, 64]" = torch.ops.aten.view.default(permute_2, _shape_param_4);  permute_2 = _shape_param_4 = None
        add_1: "f32[512, 16, 16, 64]" = torch.ops.aten.add.Tensor(view_2, arg3_1);  view_2 = arg3_1 = None
        convert_element_type_2: "bf16[512, 16, 16, 64]" = torch.ops.prims.convert_element_type.default(add_1, torch.bfloat16);  add_1 = None
        unsqueeze_4: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.view.default(convert_element_type_2, _shape_param_5);  convert_element_type_2 = _shape_param_5 = None
        permute_3: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_4, [1, 2, 0, 4, 3]);  unsqueeze_4 = None
        permute_4: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_3, [0, 1, 2, 4, 3]);  permute_3 = None
        view_4: "bf16[256, 512, 64]" = torch.ops.aten.view.default(permute_4, _shape_param_6);  permute_4 = _shape_param_6 = None
        permute_5: "bf16[256, 64, 512]" = torch.ops.aten.permute.default(view_4, [0, 2, 1])
        permute_6: "bf16[256, 64, 512]" = torch.ops.aten.permute.default(view_3, [0, 2, 1])
        return (view_3, view_4, permute_5, permute_6)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
