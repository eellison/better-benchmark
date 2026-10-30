"""Curated MM repro. Pattern hash: 5ed4ad44f857."""
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
    def forward(self, arg0_1: "bf16[32768, 512]", arg1_1: "bf16[512, 128]", arg2_1: "f32[256, 128, 128]", arg3_1: "bf16[32768, 128]", arg4_1: "f32[128]", arg5_1: "f32[128]", arg6_1: "bf16[32768, 128]", arg7_1: "f32[128]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        mm: "bf16[32768, 128]" = torch.ops.aten.mm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        view: "bf16[256, 128, 128]" = torch.ops.aten.view.default(mm, _shape_param_0);  mm = _shape_param_0 = None
        convert_element_type: "f32[256, 128, 128]" = torch.ops.prims.convert_element_type.default(view, torch.float32);  view = None
        add: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(arg2_1, convert_element_type);  arg2_1 = convert_element_type = None
        view_1: "bf16[256, 128, 128]" = torch.ops.aten.view.default(arg3_1, _shape_param_1);  arg3_1 = _shape_param_1 = None
        mul: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_1, arg4_1);  view_1 = arg4_1 = None
        add_1: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul, arg5_1);  mul = arg5_1 = None
        view_2: "bf16[256, 128, 128]" = torch.ops.aten.view.default(arg6_1, _shape_param_2);  arg6_1 = _shape_param_2 = None
        add_2: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_2, add_1);  view_2 = add_1 = None
        mul_1: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add, add_2);  add_2 = None
        mul_2: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add, arg7_1);  arg7_1 = None
        convert_element_type_1: "bf16[256, 128, 128]" = torch.ops.prims.convert_element_type.default(mul_2, torch.bfloat16)
        return (add, mul_1, mul_2, convert_element_type_1)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
