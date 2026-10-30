"""Curated MM repro. Pattern hash: 3b62e129cb7f."""
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
    def forward(self, arg0_1: "f32[1024, 1024]", arg1_1: "f32[64, 256, 1024]", arg2_1: "bf16[1024]", arg3_1: "f32[1024, 1024]", arg4_1: "bf16[1024]", arg5_1: "f32[1024, 1024]", arg6_1: "bf16[1024]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7):
        # No stacktrace found for following nodes
        convert_element_type: "bf16[1024, 1024]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.bfloat16);  arg0_1 = None
        convert_element_type_1: "bf16[64, 256, 1024]" = torch.ops.prims.convert_element_type.default(arg1_1, torch.bfloat16);  arg1_1 = None
        view: "bf16[16384, 1024]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_0);  convert_element_type_1 = _shape_param_0 = None
        addmm: "bf16[16384, 1024]" = torch.ops.aten.addmm.default(arg2_1, view, convert_element_type);  arg2_1 = convert_element_type = None
        view_1: "bf16[64, 256, 1024]" = torch.ops.aten.view.default(addmm, _shape_param_1);  addmm = _shape_param_1 = None
        mul: "bf16[64, 256, 1024]" = torch.ops.aten.mul.Tensor(view_1, 0.125);  view_1 = None
        convert_element_type_2: "bf16[1024, 1024]" = torch.ops.prims.convert_element_type.default(arg3_1, torch.bfloat16);  arg3_1 = None
        addmm_1: "bf16[16384, 1024]" = torch.ops.aten.addmm.default(arg4_1, view, convert_element_type_2);  arg4_1 = convert_element_type_2 = None
        convert_element_type_3: "bf16[1024, 1024]" = torch.ops.prims.convert_element_type.default(arg5_1, torch.bfloat16);  arg5_1 = None
        addmm_2: "bf16[16384, 1024]" = torch.ops.aten.addmm.default(arg6_1, view, convert_element_type_3);  arg6_1 = view = convert_element_type_3 = None
        view_4: "bf16[64, 256, 16, 64]" = torch.ops.aten.view.default(addmm_1, _shape_param_2);  addmm_1 = _shape_param_2 = None
        permute_3: "bf16[64, 16, 256, 64]" = torch.ops.aten.permute.default(view_4, [0, 2, 1, 3]);  view_4 = None
        view_5: "bf16[64, 256, 16, 64]" = torch.ops.aten.view.default(addmm_2, _shape_param_3);  addmm_2 = _shape_param_3 = None
        permute_4: "bf16[64, 16, 256, 64]" = torch.ops.aten.permute.default(view_5, [0, 2, 1, 3]);  view_5 = None
        view_6: "bf16[64, 256, 16, 64]" = torch.ops.aten.view.default(mul, _shape_param_4);  mul = _shape_param_4 = None
        permute_5: "bf16[64, 16, 256, 64]" = torch.ops.aten.permute.default(view_6, [0, 2, 1, 3]);  view_6 = None
        clone: "bf16[64, 16, 256, 64]" = torch.ops.aten.clone.default(permute_5, memory_format = torch.contiguous_format);  permute_5 = None
        view_7: "bf16[1024, 256, 64]" = torch.ops.aten.view.default(clone, _shape_param_5);  clone = _shape_param_5 = None
        clone_1: "bf16[64, 16, 256, 64]" = torch.ops.aten.clone.default(permute_3, memory_format = torch.contiguous_format);  permute_3 = None
        view_8: "bf16[1024, 256, 64]" = torch.ops.aten.view.default(clone_1, _shape_param_6);  clone_1 = _shape_param_6 = None
        clone_2: "bf16[64, 16, 256, 64]" = torch.ops.aten.clone.default(permute_4, memory_format = torch.contiguous_format);  permute_4 = None
        view_9: "bf16[1024, 256, 64]" = torch.ops.aten.view.default(clone_2, _shape_param_7);  clone_2 = _shape_param_7 = None
        permute_6: "bf16[1024, 64, 256]" = torch.ops.aten.permute.default(view_8, [0, 2, 1]);  view_8 = None
        permute_7: "bf16[1024, 64, 256]" = torch.ops.aten.permute.default(view_9, [0, 2, 1])
        return (view_7, view_9, permute_6, permute_7)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
