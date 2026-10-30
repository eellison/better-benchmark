"""Curated MM repro. Pattern hash: 99b2103ff4db."""
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
    def forward(self, arg0_1: "bf16[128, 12, 198, 64]", arg1_1: "bf16[128, 12, 198, 64]", arg2_1: "bf16[128, 12, 198, 64]", arg3_1: "bf16[2304, 768]", arg4_1: "f32[768]", arg5_1: "f32[128, 198, 768]", arg6_1: "f32[1, 198, 768]", arg7_1: "f32[128, 198, 1]", arg8_1: "f32[128, 198, 1]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        cat: "bf16[384, 12, 198, 64]" = torch.ops.aten.cat.default([arg0_1, arg1_1, arg2_1]);  arg0_1 = arg1_1 = arg2_1 = None
        view: "bf16[3, 128, 12, 198, 64]" = torch.ops.aten.view.default(cat, _shape_param_0);  cat = _shape_param_0 = None
        permute: "bf16[128, 198, 3, 12, 64]" = torch.ops.aten.permute.default(view, [1, 3, 0, 2, 4]);  view = None
        clone: "bf16[128, 198, 3, 12, 64]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format);  permute = None
        view_2: "bf16[25344, 2304]" = torch.ops.aten.view.default(clone, _shape_param_1);  clone = _shape_param_1 = None
        mm: "bf16[25344, 768]" = torch.ops.aten.mm.default(view_2, arg3_1);  view_2 = arg3_1 = None
        view_3: "bf16[128, 198, 768]" = torch.ops.aten.view.default(mm, _shape_param_2);  mm = _shape_param_2 = None
        convert_element_type: "f32[128, 198, 768]" = torch.ops.prims.convert_element_type.default(view_3, torch.float32);  view_3 = None
        mul: "f32[128, 198, 768]" = torch.ops.aten.mul.Tensor(convert_element_type, arg4_1);  arg4_1 = None
        mul_1: "f32[128, 198, 768]" = torch.ops.aten.mul.Tensor(mul, 768)
        add: "f32[128, 198, 768]" = torch.ops.aten.add.Tensor(arg5_1, arg6_1);  arg5_1 = arg6_1 = None
        sub: "f32[128, 198, 768]" = torch.ops.aten.sub.Tensor(add, arg7_1);  add = arg7_1 = None
        mul_2: "f32[128, 198, 768]" = torch.ops.aten.mul.Tensor(sub, arg8_1);  sub = arg8_1 = None
        mul_3: "f32[128, 198, 768]" = torch.ops.aten.mul.Tensor(convert_element_type, mul_2);  mul_2 = None
        return (convert_element_type, mul, mul_1, mul_3)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
