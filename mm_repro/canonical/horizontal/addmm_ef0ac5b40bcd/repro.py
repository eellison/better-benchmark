"""Curated MM repro. Pattern hash: ef0ac5b40bcd."""
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
    def forward(self, arg0_1: "f32[4096, 4096]", arg1_1: "bf16[4096]", arg2_1: "bf16[4096, 4096]", arg3_1: "f32[4096, 4096]", arg4_1: "bf16[4096]", arg5_1: "f32[4096, 4096]", arg6_1: "bf16[4096]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4):
        # No stacktrace found for following nodes
        convert_element_type: "bf16[4096, 4096]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.bfloat16);  arg0_1 = None
        addmm: "bf16[4096, 4096]" = torch.ops.aten.addmm.default(arg1_1, arg2_1, convert_element_type);  arg1_1 = convert_element_type = None
        view_1: "bf16[8, 512, 64, 64]" = torch.ops.aten.view.default(addmm, _shape_param_0);  addmm = _shape_param_0 = None
        permute_1: "bf16[8, 64, 512, 64]" = torch.ops.aten.permute.default(view_1, [0, 2, 1, 3]);  view_1 = None
        convert_element_type_1: "bf16[4096, 4096]" = torch.ops.prims.convert_element_type.default(arg3_1, torch.bfloat16);  arg3_1 = None
        addmm_1: "bf16[4096, 4096]" = torch.ops.aten.addmm.default(arg4_1, arg2_1, convert_element_type_1);  arg4_1 = convert_element_type_1 = None
        view_3: "bf16[8, 512, 64, 64]" = torch.ops.aten.view.default(addmm_1, _shape_param_1);  addmm_1 = _shape_param_1 = None
        permute_3: "bf16[8, 64, 512, 64]" = torch.ops.aten.permute.default(view_3, [0, 2, 1, 3]);  view_3 = None
        convert_element_type_2: "bf16[4096, 4096]" = torch.ops.prims.convert_element_type.default(arg5_1, torch.bfloat16);  arg5_1 = None
        addmm_2: "bf16[4096, 4096]" = torch.ops.aten.addmm.default(arg6_1, arg2_1, convert_element_type_2);  arg6_1 = arg2_1 = convert_element_type_2 = None
        view_5: "bf16[8, 512, 64, 64]" = torch.ops.aten.view.default(addmm_2, _shape_param_2);  addmm_2 = _shape_param_2 = None
        permute_5: "bf16[8, 64, 512, 64]" = torch.ops.aten.permute.default(view_5, [0, 2, 1, 3]);  view_5 = None
        mul: "bf16[8, 64, 512, 64]" = torch.ops.aten.mul.Scalar(permute_1, 0.3535533905932738);  permute_1 = None
        permute_6: "bf16[8, 64, 64, 512]" = torch.ops.aten.permute.default(permute_3, [0, 1, 3, 2]);  permute_3 = None
        mul_1: "bf16[8, 64, 64, 512]" = torch.ops.aten.mul.Scalar(permute_6, 0.3535533905932738);  permute_6 = None
        clone: "bf16[8, 64, 512, 64]" = torch.ops.aten.clone.default(mul, memory_format = torch.contiguous_format);  mul = None
        view_6: "bf16[512, 512, 64]" = torch.ops.aten.view.default(clone, _shape_param_3);  clone = _shape_param_3 = None
        clone_1: "bf16[8, 64, 64, 512]" = torch.ops.aten.clone.default(mul_1, memory_format = torch.contiguous_format);  mul_1 = None
        clone_2: "bf16[8, 64, 512, 64]" = torch.ops.aten.clone.default(permute_5, memory_format = torch.contiguous_format);  permute_5 = None
        view_7: "bf16[512, 512, 64]" = torch.ops.aten.view.default(clone_2, _shape_param_4);  clone_2 = _shape_param_4 = None
        permute_7: "bf16[512, 64, 512]" = torch.ops.aten.permute.default(view_7, [0, 2, 1])
        return (view_6, clone_1, view_7, permute_7)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
