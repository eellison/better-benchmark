"""Curated MM repro. Pattern hash: 83df85014e51."""
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
    def forward(self, arg0_1: "f32[512, 256, 144]", arg1_1: "f32[512, 256, 144]", arg2_1: "f32[512, 256, 144]", arg3_1: "f32[512, 256, 144]", arg4_1: "f32[512, 256, 1]", arg5_1: "bf16[512, 256, 144]", arg6_1: "bf16[144, 288]", arg7_1: "bf16[131072, 288]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # No stacktrace found for following nodes
        sum_1: "f32[512, 256, 1]" = torch.ops.aten.sum.dim_IntList(arg0_1, [2], True);  arg0_1 = None
        sum_2: "f32[512, 256, 1]" = torch.ops.aten.sum.dim_IntList(arg1_1, [2], True);  arg1_1 = None
        mul: "f32[512, 256, 144]" = torch.ops.aten.mul.Tensor(arg2_1, sum_2);  arg2_1 = sum_2 = None
        sub: "f32[512, 256, 144]" = torch.ops.aten.sub.Tensor(arg3_1, sum_1);  arg3_1 = sum_1 = None
        sub_1: "f32[512, 256, 144]" = torch.ops.aten.sub.Tensor(sub, mul);  sub = mul = None
        div: "f32[512, 256, 1]" = torch.ops.aten.div.Tensor(arg4_1, 144);  arg4_1 = None
        mul_1: "f32[512, 256, 144]" = torch.ops.aten.mul.Tensor(div, sub_1);  div = sub_1 = None
        convert_element_type: "bf16[512, 256, 144]" = torch.ops.prims.convert_element_type.default(mul_1, torch.bfloat16);  mul_1 = None
        add: "bf16[512, 256, 144]" = torch.ops.aten.add.Tensor(arg5_1, convert_element_type);  arg5_1 = convert_element_type = None
        view: "bf16[131072, 144]" = torch.ops.aten.view.default(add, _shape_param_0);  _shape_param_0 = None
        mm: "bf16[131072, 288]" = torch.ops.aten.mm.default(view, arg6_1);  arg6_1 = None
        permute: "bf16[144, 131072]" = torch.ops.aten.permute.default(view, [1, 0])
        view_1: "bf16[512, 256, 288]" = torch.ops.aten.view.default(mm, _shape_param_1);  mm = _shape_param_1 = None
        convert_element_type_1: "f32[512, 256, 288]" = torch.ops.prims.convert_element_type.default(view_1, torch.float32);  view_1 = None
        view_2: "bf16[512, 256, 288]" = torch.ops.aten.view.default(arg7_1, _shape_param_2);  arg7_1 = _shape_param_2 = None
        convert_element_type_2: "f32[512, 256, 288]" = torch.ops.prims.convert_element_type.default(view_2, torch.float32);  view_2 = None
        sigmoid: "f32[512, 256, 288]" = torch.ops.aten.sigmoid.default(convert_element_type_2)
        mul_2: "f32[512, 256, 288]" = torch.ops.aten.mul.Tensor(convert_element_type_1, sigmoid);  convert_element_type_1 = None
        sub_2: "f32[512, 256, 288]" = torch.ops.aten.sub.Tensor(1, sigmoid);  sigmoid = None
        mul_3: "f32[512, 256, 288]" = torch.ops.aten.mul.Tensor(convert_element_type_2, sub_2);  convert_element_type_2 = sub_2 = None
        add_1: "f32[512, 256, 288]" = torch.ops.aten.add.Tensor(mul_3, 1);  mul_3 = None
        mul_4: "f32[512, 256, 288]" = torch.ops.aten.mul.Tensor(mul_2, add_1);  mul_2 = add_1 = None
        convert_element_type_3: "bf16[512, 256, 288]" = torch.ops.prims.convert_element_type.default(mul_4, torch.bfloat16);  mul_4 = None
        view_3: "bf16[131072, 288]" = torch.ops.aten.view.default(convert_element_type_3, _shape_param_3);  convert_element_type_3 = _shape_param_3 = None
        return (add, view, permute, view_3)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
