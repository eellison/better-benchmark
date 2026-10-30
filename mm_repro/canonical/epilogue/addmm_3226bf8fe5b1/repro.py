"""Curated MM repro. Pattern hash: 3226bf8fe5b1."""
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
    def forward(self, arg0_1: "f32[2560]", arg1_1: "f32[2560, 2560]", arg2_1: "f32[16, 128, 2560]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        convert_element_type: "bf16[2560]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.bfloat16);  arg0_1 = None
        convert_element_type_1: "bf16[2560, 2560]" = torch.ops.prims.convert_element_type.default(arg1_1, torch.bfloat16);  arg1_1 = None
        convert_element_type_2: "bf16[16, 128, 2560]" = torch.ops.prims.convert_element_type.default(arg2_1, torch.bfloat16);  arg2_1 = None
        view: "bf16[2048, 2560]" = torch.ops.aten.view.default(convert_element_type_2, _shape_param_0);  convert_element_type_2 = _shape_param_0 = None
        addmm: "bf16[2048, 2560]" = torch.ops.aten.addmm.default(convert_element_type, view, convert_element_type_1);  convert_element_type = view = convert_element_type_1 = None
        view_2: "bf16[16, 128, 32, 80]" = torch.ops.aten.view.default(addmm, _shape_param_1);  addmm = _shape_param_1 = None
        permute_1: "bf16[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_2, [0, 2, 1, 3]);  view_2 = None
        permute_2: "bf16[16, 32, 80, 128]" = torch.ops.aten.permute.default(permute_1, [0, 1, 3, 2]);  permute_1 = None
        mul: "bf16[16, 32, 80, 128]" = torch.ops.aten.mul.Scalar(permute_2, 0.334370152488211);  permute_2 = None
        clone: "bf16[16, 32, 80, 128]" = torch.ops.aten.clone.default(mul, memory_format = torch.contiguous_format);  mul = None
        view_3: "bf16[512, 80, 128]" = torch.ops.aten.view.default(clone, _shape_param_2);  clone = _shape_param_2 = None
        permute_3: "bf16[512, 128, 80]" = torch.ops.aten.permute.default(view_3, [0, 2, 1])
        return (view_3, permute_3)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
