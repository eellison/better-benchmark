"""Curated MM repro. Pattern hash: ff0bb3799d9c."""
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
    def forward(self, arg0_1: "f32[128]", arg1_1: "f32[128, 128]", arg2_1: "bf16[32768, 128]", arg3_1: "bf16[1024, 128, 32]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        convert_element_type: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.bfloat16);  arg0_1 = None
        convert_element_type_1: "bf16[128, 128]" = torch.ops.prims.convert_element_type.default(arg1_1, torch.bfloat16);  arg1_1 = None
        addmm: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type, arg2_1, convert_element_type_1);  convert_element_type = arg2_1 = convert_element_type_1 = None
        view_1: "bf16[256, 128, 4, 32]" = torch.ops.aten.view.default(addmm, _shape_param_0);  addmm = _shape_param_0 = None
        permute_1: "bf16[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_1, [0, 2, 1, 3]);  view_1 = None
        permute_2: "bf16[256, 4, 32, 128]" = torch.ops.aten.permute.default(permute_1, [0, 1, 3, 2]);  permute_1 = None
        mul: "bf16[256, 4, 32, 128]" = torch.ops.aten.mul.Scalar(permute_2, 0.4204482076268573);  permute_2 = None
        clone: "bf16[256, 4, 32, 128]" = torch.ops.aten.clone.default(mul, memory_format = torch.contiguous_format);  mul = None
        view_2: "bf16[1024, 32, 128]" = torch.ops.aten.view.default(clone, _shape_param_1);  clone = _shape_param_1 = None
        bmm: "bf16[1024, 128, 128]" = torch.ops.aten.bmm.default(arg3_1, view_2);  arg3_1 = None
        view_3: "bf16[256, 4, 128, 128]" = torch.ops.aten.view.default(bmm, _shape_param_2);  bmm = _shape_param_2 = None
        convert_element_type_2: "f32[256, 4, 128, 128]" = torch.ops.prims.convert_element_type.default(view_3, torch.float32)
        eq: "b8[256, 4, 128, 128]" = torch.ops.aten.eq.Scalar(view_3, -inf);  view_3 = None
        logical_not: "b8[256, 4, 128, 128]" = torch.ops.aten.logical_not.default(eq);  eq = None
        permute_3: "bf16[1024, 128, 32]" = torch.ops.aten.permute.default(view_2, [0, 2, 1]);  view_2 = None
        return (convert_element_type_2, logical_not, permute_3)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
