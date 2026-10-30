"""Curated MM repro. Pattern hash: 5719038c1c4a."""
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
    def forward(self, arg0_1: "bf16[32768, 30522]", arg1_1: "f32[32768, 1]", arg2_1: "f32[32768, 30522]", arg3_1: "bf16[256, 128, 30522]", arg4_1: "bf16[512, 32768]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        convert_element_type: "f32[32768, 30522]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.float32);  arg0_1 = None
        exp: "f32[32768, 30522]" = torch.ops.aten.exp.default(convert_element_type);  convert_element_type = None
        mul: "f32[32768, 30522]" = torch.ops.aten.mul.Tensor(exp, arg1_1);  exp = arg1_1 = None
        sub: "f32[32768, 30522]" = torch.ops.aten.sub.Tensor(arg2_1, mul);  arg2_1 = mul = None
        convert_element_type_1: "bf16[32768, 30522]" = torch.ops.prims.convert_element_type.default(sub, torch.bfloat16);  sub = None
        view: "bf16[256, 128, 30522]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_0);  convert_element_type_1 = _shape_param_0 = None
        add: "bf16[256, 128, 30522]" = torch.ops.aten.add.Tensor(arg3_1, view);  arg3_1 = view = None
        view_1: "bf16[32768, 30522]" = torch.ops.aten.view.default(add, _shape_param_1);  add = _shape_param_1 = None
        constant_pad_nd: "bf16[32768, 30528]" = torch.ops.aten.constant_pad_nd.default(view_1, _shape_param_2);  view_1 = _shape_param_2 = None
        mm: "bf16[512, 30528]" = torch.ops.aten.mm.default(arg4_1, constant_pad_nd);  arg4_1 = constant_pad_nd = None
        slice_1: "bf16[512, 30522]" = torch.ops.aten.slice.Tensor(mm, 1, 0, -6);  mm = None
        convert_element_type_2: "f32[512, 30522]" = torch.ops.prims.convert_element_type.default(slice_1, torch.float32);  slice_1 = None
        slice_2: "f32[128, 30522]" = torch.ops.aten.slice.Tensor(convert_element_type_2, 0, 0, 128)
        slice_3: "f32[384, 30522]" = torch.ops.aten.slice.Tensor(convert_element_type_2, 0, 128, 512);  convert_element_type_2 = None
        permute: "f32[30522, 128]" = torch.ops.aten.permute.default(slice_2, [1, 0]);  slice_2 = None
        return (slice_3, permute)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
