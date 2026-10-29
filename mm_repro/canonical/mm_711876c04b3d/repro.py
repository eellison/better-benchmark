"""Curated MM repro. Pattern hash: 711876c04b3d."""
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
    def forward(self, arg0_1: "bf16[96, 65]", arg1_1: "bf16[96, 65]", arg2_1: "bf16[65, 512]", arg3_1: "bf16[96, 512, 4, 4]", arg4_1: "b8[96, 512, 4, 4]", arg5_1: "bf16[]", _shape_param_0):
        # No stacktrace found for following nodes
        convert_element_type: "f32[96, 65]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.float32);  arg0_1 = None
        convert_element_type_1: "f32[96, 65]" = torch.ops.prims.convert_element_type.default(arg1_1, torch.float32);  arg1_1 = None
        sub: "f32[96, 65]" = torch.ops.aten.sub.Tensor(1, convert_element_type_1)
        mul: "f32[96, 65]" = torch.ops.aten.mul.Tensor(convert_element_type_1, sub);  convert_element_type_1 = sub = None
        mul_1: "f32[96, 65]" = torch.ops.aten.mul.Tensor(convert_element_type, mul);  convert_element_type = mul = None
        convert_element_type_2: "bf16[96, 65]" = torch.ops.prims.convert_element_type.default(mul_1, torch.bfloat16);  mul_1 = None
        mm: "bf16[96, 512]" = torch.ops.aten.mm.default(convert_element_type_2, arg2_1);  convert_element_type_2 = arg2_1 = None
        view: "bf16[96, 512, 1, 1]" = torch.ops.aten.view.default(mm, _shape_param_0);  mm = _shape_param_0 = None
        avg_pool2d_backward: "bf16[96, 512, 4, 4]" = torch.ops.aten.avg_pool2d_backward.default(view, arg3_1, [4, 4], [], [0, 0], False, True, None);  view = arg3_1 = None
        where: "bf16[96, 512, 4, 4]" = torch.ops.aten.where.self(arg4_1, arg5_1, avg_pool2d_backward);  arg4_1 = arg5_1 = avg_pool2d_backward = None
        convert_element_type_3: "f32[96, 512, 4, 4]" = torch.ops.prims.convert_element_type.default(where, torch.float32)
        return (where, convert_element_type_3)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
