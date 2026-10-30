"""Curated MM repro. Pattern hash: e8d7b5bc99fa."""
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
    def forward(self, arg0_1: "f32[128, 3136, 128]", arg1_1: "f32[128, 3136, 128]", arg2_1: "f32[128, 3136, 128]", arg3_1: "f32[128, 3136, 128]", arg4_1: "f32[128, 3136, 1]", arg5_1: "f32[128, 3136, 128]", arg6_1: "bf16[128, 128]", arg7_1: "bf16[32768, 32, 49]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4):
        # No stacktrace found for following nodes
        sum_1: "f32[128, 3136, 1]" = torch.ops.aten.sum.dim_IntList(arg0_1, [2], True);  arg0_1 = None
        sum_2: "f32[128, 3136, 1]" = torch.ops.aten.sum.dim_IntList(arg1_1, [2], True);  arg1_1 = None
        mul: "f32[128, 3136, 128]" = torch.ops.aten.mul.Tensor(arg2_1, sum_2);  arg2_1 = sum_2 = None
        sub: "f32[128, 3136, 128]" = torch.ops.aten.sub.Tensor(arg3_1, sum_1);  arg3_1 = sum_1 = None
        sub_1: "f32[128, 3136, 128]" = torch.ops.aten.sub.Tensor(sub, mul);  sub = mul = None
        mul_1: "f32[128, 3136, 128]" = torch.ops.aten.mul.Tensor(arg4_1, sub_1);  arg4_1 = sub_1 = None
        add: "f32[128, 3136, 128]" = torch.ops.aten.add.Tensor(arg5_1, mul_1);  arg5_1 = mul_1 = None
        view: "f32[128, 56, 56, 128]" = torch.ops.aten.view.default(add, _shape_param_0);  add = _shape_param_0 = None
        convert_element_type: "bf16[128, 56, 56, 128]" = torch.ops.prims.convert_element_type.default(view, torch.bfloat16)
        view_1: "bf16[128, 8, 7, 8, 7, 128]" = torch.ops.aten.view.default(convert_element_type, _shape_param_1);  convert_element_type = _shape_param_1 = None
        permute: "bf16[128, 8, 8, 7, 7, 128]" = torch.ops.aten.permute.default(view_1, [0, 1, 3, 2, 4, 5]);  view_1 = None
        clone: "bf16[128, 8, 8, 7, 7, 128]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format);  permute = None
        view_4: "bf16[401408, 128]" = torch.ops.aten.view.default(clone, _shape_param_2);  clone = _shape_param_2 = None
        mm: "bf16[401408, 128]" = torch.ops.aten.mm.default(view_4, arg6_1);  arg6_1 = None
        view_6: "bf16[8192, 49, 4, 32]" = torch.ops.aten.view.default(mm, _shape_param_3);  mm = _shape_param_3 = None
        permute_1: "bf16[8192, 4, 49, 32]" = torch.ops.aten.permute.default(view_6, [0, 2, 1, 3]);  view_6 = None
        clone_1: "bf16[8192, 4, 49, 32]" = torch.ops.aten.clone.default(permute_1, memory_format = torch.contiguous_format);  permute_1 = None
        view_7: "bf16[32768, 49, 32]" = torch.ops.aten.view.default(clone_1, _shape_param_4);  clone_1 = _shape_param_4 = None
        bmm: "bf16[32768, 49, 49]" = torch.ops.aten.bmm.default(view_7, arg7_1);  arg7_1 = None
        return (view, view_4, view_7, bmm)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
