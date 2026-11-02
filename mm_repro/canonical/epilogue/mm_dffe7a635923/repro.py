"""Curated MM repro. Pattern hash: dffe7a635923."""
import sys
from pathlib import Path
from math import inf, nan
import torch
import torch._inductor.inductor_prims
from torch import device
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from repro_harness import benchmark_repro, load_shape_configs, make_inputs_from_config
from mm_replay import make_inputs_from_config, benchmark_repro_safe as benchmark_repro
_repro_version = 3

class Repro(torch.nn.Module):
    def forward(self, arg0_1: "bf16[128, 16384]", arg1_1: "bf16[16384, 4096]", arg2_1: "bf16[128, 4096]", arg3_1: "bf16[128, 4096]", arg4_1: "bf16[128, 4096]", arg5_1: "f32[4096]", arg6_1: "f32[1, 128, 4096]", arg7_1: "f32[1, 128, 1]", arg8_1: "f32[1, 128, 4096]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5):
        # No stacktrace found for following nodes
        mm: "bf16[128, 4096]" = torch.ops.aten.mm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        view: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm, _shape_param_0);  mm = _shape_param_0 = None
        convert_element_type: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view, torch.float32);  view = None
        view_1: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(arg2_1, _shape_param_1);  arg2_1 = _shape_param_1 = None
        convert_element_type_1: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_1, torch.float32);  view_1 = None
        add: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(convert_element_type, convert_element_type_1);  convert_element_type = convert_element_type_1 = None
        view_2: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(arg3_1, _shape_param_2);  arg3_1 = _shape_param_2 = None
        convert_element_type_2: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_2, torch.float32);  view_2 = None
        add_1: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add, convert_element_type_2);  add = convert_element_type_2 = None
        view_3: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(arg4_1, _shape_param_3);  arg4_1 = _shape_param_3 = None
        convert_element_type_3: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_3, torch.float32);  view_3 = None
        add_2: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_1, convert_element_type_3);  add_1 = convert_element_type_3 = None
        mul: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_2, arg5_1);  arg5_1 = None
        mul_1: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul, 4096)
        sum_1: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul, [2], True)
        mul_2: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul, arg6_1);  mul = None
        sum_2: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_2, [2], True);  mul_2 = None
        mul_3: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(arg6_1, sum_2);  sum_2 = None
        sub: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(mul_1, sum_1);  mul_1 = sum_1 = None
        sub_1: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(sub, mul_3);  sub = mul_3 = None
        mul_4: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(arg7_1, sub_1);  arg7_1 = sub_1 = None
        mul_5: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_2, arg6_1);  arg6_1 = None
        sum_3: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_5, [0, 1]);  mul_5 = None
        sum_4: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_2, [0, 1]);  add_2 = None
        add_3: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(arg8_1, mul_4);  arg8_1 = mul_4 = None
        convert_element_type_4: "bf16[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(add_3, torch.bfloat16)
        view_4: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_4, _shape_param_4);  convert_element_type_4 = _shape_param_4 = None
        permute: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_4, [1, 0])
        sum_5: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_4, [0], True, dtype = torch.float32)
        view_5: "f32[4096]" = torch.ops.aten.view.default(sum_5, _shape_param_5);  sum_5 = _shape_param_5 = None
        convert_element_type_5: "bf16[4096]" = torch.ops.prims.convert_element_type.default(view_5, torch.bfloat16);  view_5 = None
        convert_element_type_6: "f32[4096]" = torch.ops.prims.convert_element_type.default(convert_element_type_5, torch.float32);  convert_element_type_5 = None
        return (sum_3, sum_4, add_3, view_4, permute, convert_element_type_6)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
