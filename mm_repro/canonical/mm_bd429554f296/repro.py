"""Curated MM repro. Pattern hash: bd429554f296."""
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
    def forward(self, arg0_1: "f32[16, 128, 20005]", arg1_1: "f32[16, 128, 1]", arg2_1: "f32[16, 128, 20005]", arg3_1: "bf16[20005, 768]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # No stacktrace found for following nodes
        exp: "f32[16, 128, 20005]" = torch.ops.aten.exp.default(arg0_1);  arg0_1 = None
        mul: "f32[16, 128, 20005]" = torch.ops.aten.mul.Tensor(exp, arg1_1);  exp = arg1_1 = None
        sub: "f32[16, 128, 20005]" = torch.ops.aten.sub.Tensor(arg2_1, mul);  arg2_1 = mul = None
        convert_element_type: "bf16[16, 128, 20005]" = torch.ops.prims.convert_element_type.default(sub, torch.bfloat16);  sub = None
        view: "bf16[2048, 20005]" = torch.ops.aten.view.default(convert_element_type, _shape_param_0);  convert_element_type = _shape_param_0 = None
        constant_pad_nd: "bf16[2048, 20008]" = torch.ops.aten.constant_pad_nd.default(view, _shape_param_1);  view = _shape_param_1 = None
        constant_pad_nd_1: "bf16[20008, 768]" = torch.ops.aten.constant_pad_nd.default(arg3_1, _shape_param_2);  arg3_1 = _shape_param_2 = None
        mm: "bf16[2048, 768]" = torch.ops.aten.mm.default(constant_pad_nd, constant_pad_nd_1);  constant_pad_nd = constant_pad_nd_1 = None
        view_1: "bf16[16, 128, 768]" = torch.ops.aten.view.default(mm, _shape_param_3);  mm = _shape_param_3 = None
        convert_element_type_1: "f32[16, 128, 768]" = torch.ops.prims.convert_element_type.default(view_1, torch.float32);  view_1 = None
        return convert_element_type_1

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
