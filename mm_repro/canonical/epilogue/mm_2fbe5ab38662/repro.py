"""Curated MM repro. Pattern hash: 2fbe5ab38662."""
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
    def forward(self, arg0_1: "bf16[4096, 16384]", arg1_1: "bf16[16384, 4096]", arg2_1: "f32[8, 512, 4096]", arg3_1: "f32[4096]", arg4_1: "f32[8, 512, 4096]", arg5_1: "f32[8, 512, 1]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        mm: "bf16[4096, 4096]" = torch.ops.aten.mm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        view: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(mm, _shape_param_0);  mm = _shape_param_0 = None
        convert_element_type: "f32[8, 512, 4096]" = torch.ops.prims.convert_element_type.default(view, torch.float32);  view = None
        add: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(arg2_1, convert_element_type);  arg2_1 = convert_element_type = None
        mul: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add, arg3_1);  arg3_1 = None
        mul_1: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul, 4096)
        sum_1: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul, [2], True)
        mul_2: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul, arg4_1);  mul = None
        sum_2: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_2, [2], True);  mul_2 = None
        mul_3: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(arg4_1, sum_2);  arg4_1 = sum_2 = None
        sub: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(mul_1, sum_1);  mul_1 = sum_1 = None
        sub_1: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(sub, mul_3);  sub = mul_3 = None
        mul_4: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(arg5_1, sub_1);  arg5_1 = sub_1 = None
        convert_element_type_1: "bf16[8, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_4, torch.bfloat16)
        view_1: "bf16[4096, 4096]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_1);  convert_element_type_1 = _shape_param_1 = None
        permute: "bf16[4096, 4096]" = torch.ops.aten.permute.default(view_1, [1, 0])
        return (add, mul_4, view_1, permute)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
