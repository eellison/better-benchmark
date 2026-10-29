"""Curated MM repro. Pattern hash: a94c92c04f7c."""
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
    def forward(self, arg0_1: "f32[1024, 16, 64]", arg1_1: "bf16[8192, 1024]", arg2_1: "b8[512, 16, 1024]", arg3_1: "f32[512, 16, 1024]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # No stacktrace found for following nodes
        convert_element_type: "bf16[1024, 16, 64]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.bfloat16);  arg0_1 = None
        unsqueeze_1: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.view.default(convert_element_type, _shape_param_0);  convert_element_type = _shape_param_0 = None
        permute: "bf16[1, 1, 1024, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_1, [3, 4, 0, 2, 1]);  unsqueeze_1 = None
        permute_1: "bf16[64, 16, 1024, 1, 1]" = torch.ops.aten.permute.default(permute, [3, 4, 2, 0, 1]);  permute = None
        clone: "bf16[64, 16, 1024, 1, 1]" = torch.ops.aten.clone.default(permute_1, memory_format = torch.contiguous_format);  permute_1 = None
        squeeze: "bf16[1024, 1024]" = torch.ops.aten.view.default(clone, _shape_param_1);  clone = _shape_param_1 = None
        mm: "bf16[8192, 1024]" = torch.ops.aten.mm.default(arg1_1, squeeze);  arg1_1 = squeeze = None
        view_1: "bf16[512, 16, 1, 1, 1024]" = torch.ops.aten.view.default(mm, _shape_param_2);  mm = _shape_param_2 = None
        permute_2: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.permute.default(view_1, [0, 1, 4, 2, 3]);  view_1 = None
        view_2: "bf16[512, 16, 1024]" = torch.ops.aten.view.default(permute_2, _shape_param_3);  permute_2 = _shape_param_3 = None
        mul: "bf16[512, 16, 1024]" = torch.ops.aten.mul.Tensor(arg2_1, view_2);  arg2_1 = view_2 = None
        mul_1: "bf16[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul, 1.1111111111111112);  mul = None
        add: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_1, arg3_1);  mul_1 = arg3_1 = None
        return add

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
