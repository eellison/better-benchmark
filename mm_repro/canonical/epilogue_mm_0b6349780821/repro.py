"""Curated MM repro. Pattern hash: 0b6349780821."""
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
    def forward(self, arg0_1: "bf16[32768, 512]", arg1_1: "bf16[512, 128]", arg2_1: "f32[256, 128, 128]", arg3_1: "f32[256, 128, 128]", arg4_1: "f32[128]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4):
        # No stacktrace found for following nodes
        mm: "bf16[32768, 128]" = torch.ops.aten.mm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        view: "bf16[256, 128, 128]" = torch.ops.aten.view.default(mm, _shape_param_0);  mm = _shape_param_0 = None
        convert_element_type: "f32[256, 128, 128]" = torch.ops.prims.convert_element_type.default(view, torch.float32);  view = None
        add: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(arg2_1, convert_element_type);  arg2_1 = convert_element_type = None
        sum_1: "f32[1, 1, 128]" = torch.ops.aten.sum.dim_IntList(add, [0, 1], True, dtype = torch.float32)
        view_1: "f32[128]" = torch.ops.aten.view.default(sum_1, _shape_param_1);  sum_1 = _shape_param_1 = None
        mul: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add, arg3_1);  arg3_1 = None
        mul_1: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add, arg4_1);  add = arg4_1 = None
        sum_2: "f32[1, 1, 128]" = torch.ops.aten.sum.dim_IntList(mul, [0, 1], True, dtype = torch.float32);  mul = None
        view_2: "f32[128]" = torch.ops.aten.view.default(sum_2, _shape_param_2);  sum_2 = _shape_param_2 = None
        convert_element_type_1: "bf16[256, 128, 128]" = torch.ops.prims.convert_element_type.default(mul_1, torch.bfloat16)
        view_3: "bf16[32768, 128]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_3);  convert_element_type_1 = _shape_param_3 = None
        permute: "bf16[128, 32768]" = torch.ops.aten.permute.default(view_3, [1, 0])
        sum_3: "f32[1, 128]" = torch.ops.aten.sum.dim_IntList(view_3, [0], True, dtype = torch.float32)
        view_4: "f32[128]" = torch.ops.aten.view.default(sum_3, _shape_param_4);  sum_3 = _shape_param_4 = None
        convert_element_type_2: "bf16[128]" = torch.ops.prims.convert_element_type.default(view_4, torch.bfloat16);  view_4 = None
        convert_element_type_3: "f32[128]" = torch.ops.prims.convert_element_type.default(convert_element_type_2, torch.float32);  convert_element_type_2 = None
        return (view_1, mul_1, view_2, view_3, permute, convert_element_type_3)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
