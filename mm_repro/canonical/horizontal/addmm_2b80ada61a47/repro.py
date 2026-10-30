"""Curated MM repro. Pattern hash: 2b80ada61a47."""
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
    def forward(self, arg0_1: "f32[1024, 1024]", arg1_1: "f32[64, 128, 1024]", arg2_1: "bf16[1024]", arg3_1: "f32[1024, 1024]", arg4_1: "bf16[1024]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4):
        # No stacktrace found for following nodes
        convert_element_type: "bf16[1024, 1024]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.bfloat16);  arg0_1 = None
        convert_element_type_1: "bf16[64, 128, 1024]" = torch.ops.prims.convert_element_type.default(arg1_1, torch.bfloat16);  arg1_1 = None
        view: "bf16[8192, 1024]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_0);  convert_element_type_1 = _shape_param_0 = None
        addmm: "bf16[8192, 1024]" = torch.ops.aten.addmm.default(arg2_1, view, convert_element_type);  arg2_1 = convert_element_type = None
        convert_element_type_2: "bf16[1024, 1024]" = torch.ops.prims.convert_element_type.default(arg3_1, torch.bfloat16);  arg3_1 = None
        addmm_1: "bf16[8192, 1024]" = torch.ops.aten.addmm.default(arg4_1, view, convert_element_type_2);  arg4_1 = view = convert_element_type_2 = None
        view_3: "bf16[64, 128, 16, 64]" = torch.ops.aten.view.default(addmm, _shape_param_1);  addmm = _shape_param_1 = None
        permute_2: "bf16[64, 16, 128, 64]" = torch.ops.aten.permute.default(view_3, [0, 2, 1, 3]);  view_3 = None
        view_4: "bf16[64, 128, 16, 64]" = torch.ops.aten.view.default(addmm_1, _shape_param_2);  addmm_1 = _shape_param_2 = None
        permute_3: "bf16[64, 16, 128, 64]" = torch.ops.aten.permute.default(view_4, [0, 2, 1, 3]);  view_4 = None
        permute_4: "bf16[64, 16, 64, 128]" = torch.ops.aten.permute.default(permute_2, [0, 1, 3, 2]);  permute_2 = None
        mul: "bf16[64, 16, 64, 128]" = torch.ops.aten.mul.Scalar(permute_4, 0.3535533905932738);  permute_4 = None
        clone: "bf16[64, 16, 64, 128]" = torch.ops.aten.clone.default(mul, memory_format = torch.contiguous_format);  mul = None
        view_5: "bf16[1024, 64, 128]" = torch.ops.aten.view.default(clone, _shape_param_3);  clone = _shape_param_3 = None
        clone_1: "bf16[64, 16, 128, 64]" = torch.ops.aten.clone.default(permute_3, memory_format = torch.contiguous_format);  permute_3 = None
        view_6: "bf16[1024, 128, 64]" = torch.ops.aten.view.default(clone_1, _shape_param_4);  clone_1 = _shape_param_4 = None
        permute_5: "bf16[1024, 64, 128]" = torch.ops.aten.permute.default(view_6, [0, 2, 1])
        permute_6: "bf16[1024, 128, 64]" = torch.ops.aten.permute.default(view_5, [0, 2, 1])
        return (view_5, view_6, permute_5, permute_6)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
