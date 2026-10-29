"""Curated MM repro. Pattern hash: ea7d2a227aac."""
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
    def forward(self, arg0_1: "f32[16, 128, 20005]", arg1_1: "f32[16, 128, 1]", arg2_1: "f32[16, 128, 20005]", arg3_1: "bf16[2048, 768]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        exp: "f32[16, 128, 20005]" = torch.ops.aten.exp.default(arg0_1);  arg0_1 = None
        mul: "f32[16, 128, 20005]" = torch.ops.aten.mul.Tensor(exp, arg1_1);  exp = arg1_1 = None
        sub: "f32[16, 128, 20005]" = torch.ops.aten.sub.Tensor(arg2_1, mul);  arg2_1 = mul = None
        convert_element_type: "bf16[16, 128, 20005]" = torch.ops.prims.convert_element_type.default(sub, torch.bfloat16);  sub = None
        view: "bf16[2048, 20005]" = torch.ops.aten.view.default(convert_element_type, _shape_param_0);  convert_element_type = _shape_param_0 = None
        permute: "bf16[20005, 2048]" = torch.ops.aten.permute.default(view, [1, 0]);  view = None
        constant_pad_nd: "bf16[20008, 2048]" = torch.ops.aten.constant_pad_nd.default(permute, _shape_param_1);  permute = _shape_param_1 = None
        mm: "bf16[20008, 768]" = torch.ops.aten.mm.default(constant_pad_nd, arg3_1);  constant_pad_nd = arg3_1 = None
        slice_1: "bf16[20005, 768]" = torch.ops.aten.slice.Tensor(mm, 0, 0, -3);  mm = None
        convert_element_type_1: "f32[20005, 768]" = torch.ops.prims.convert_element_type.default(slice_1, torch.float32);  slice_1 = None
        return convert_element_type_1

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
