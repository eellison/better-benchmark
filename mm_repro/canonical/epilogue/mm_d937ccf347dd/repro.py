"""Curated MM repro. Pattern hash: d937ccf347dd."""
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
    def forward(self, arg0_1: "bf16[8192, 1024]", arg1_1: "bf16[1024, 1024]", arg2_1: "f32[1024]", arg3_1: "bf16[8192, 1024]", arg4_1: "b8[64, 128, 1024]", arg5_1: "f32[64, 128, 1024]", arg6_1: "f32[64, 128, 1]", arg7_1: "f32[64, 128, 1]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        mm: "bf16[8192, 1024]" = torch.ops.aten.mm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        view: "bf16[64, 128, 1024]" = torch.ops.aten.view.default(mm, _shape_param_0);  mm = _shape_param_0 = None
        convert_element_type: "f32[64, 128, 1024]" = torch.ops.prims.convert_element_type.default(view, torch.float32);  view = None
        mul: "f32[64, 128, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type, arg2_1);  arg2_1 = None
        mul_1: "f32[64, 128, 1024]" = torch.ops.aten.mul.Tensor(mul, 1024)
        view_1: "bf16[64, 128, 1024]" = torch.ops.aten.view.default(arg3_1, _shape_param_1);  arg3_1 = _shape_param_1 = None
        mul_2: "bf16[64, 128, 1024]" = torch.ops.aten.mul.Tensor(arg4_1, view_1);  arg4_1 = view_1 = None
        mul_3: "bf16[64, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_2, 1.1111111111111112);  mul_2 = None
        add: "f32[64, 128, 1024]" = torch.ops.aten.add.Tensor(arg5_1, mul_3);  arg5_1 = mul_3 = None
        sub: "f32[64, 128, 1024]" = torch.ops.aten.sub.Tensor(add, arg6_1);  add = arg6_1 = None
        mul_4: "f32[64, 128, 1024]" = torch.ops.aten.mul.Tensor(sub, arg7_1);  sub = arg7_1 = None
        mul_5: "f32[64, 128, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type, mul_4);  mul_4 = None
        return (convert_element_type, mul, mul_1, mul_5)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
