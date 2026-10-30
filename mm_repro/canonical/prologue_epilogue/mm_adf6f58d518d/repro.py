"""Curated MM repro. Pattern hash: adf6f58d518d."""
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
    def forward(self, arg0_1: "bf16[4, 2048, 12, 64]", arg1_1: "bf16[768, 768]", arg2_1: "f32[4, 2048, 768]", arg3_1: "f32[768]", arg4_1: "f32[4, 2048, 768]", arg5_1: "f32[4, 2048, 1]", arg6_1: "f32[4, 2048, 1]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        view: "bf16[4, 2048, 768]" = torch.ops.aten.view.default(arg0_1, _shape_param_0);  arg0_1 = _shape_param_0 = None
        mul: "bf16[4, 2048, 768]" = torch.ops.aten.mul.Tensor(view, 0.125);  view = None
        view_1: "bf16[8192, 768]" = torch.ops.aten.view.default(mul, _shape_param_1);  mul = _shape_param_1 = None
        mm: "bf16[8192, 768]" = torch.ops.aten.mm.default(view_1, arg1_1);  view_1 = arg1_1 = None
        view_2: "bf16[4, 2048, 768]" = torch.ops.aten.view.default(mm, _shape_param_2);  mm = _shape_param_2 = None
        convert_element_type: "f32[4, 2048, 768]" = torch.ops.prims.convert_element_type.default(view_2, torch.float32);  view_2 = None
        add: "f32[4, 2048, 768]" = torch.ops.aten.add.Tensor(arg2_1, convert_element_type);  arg2_1 = convert_element_type = None
        mul_1: "f32[4, 2048, 768]" = torch.ops.aten.mul.Tensor(add, arg3_1);  arg3_1 = None
        mul_2: "f32[4, 2048, 768]" = torch.ops.aten.mul.Tensor(mul_1, 768)
        sub: "f32[4, 2048, 768]" = torch.ops.aten.sub.Tensor(arg4_1, arg5_1);  arg4_1 = arg5_1 = None
        mul_3: "f32[4, 2048, 768]" = torch.ops.aten.mul.Tensor(sub, arg6_1);  sub = arg6_1 = None
        mul_4: "f32[4, 2048, 768]" = torch.ops.aten.mul.Tensor(add, mul_3);  mul_3 = None
        return (add, mul_1, mul_2, mul_4)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
