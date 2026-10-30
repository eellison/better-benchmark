"""Curated MM repro. Pattern hash: e7ef67550a71."""
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
    def forward(self, arg0_1: "f32[32, 128, 1]", arg1_1: "f32[32, 128, 512]", arg2_1: "f32[512]", arg3_1: "f32[512, 384]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        add: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(arg0_1, 1e-06);  arg0_1 = None
        rsqrt: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add);  add = None
        mul: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg1_1, rsqrt);  arg1_1 = rsqrt = None
        mul_1: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg2_1, mul);  arg2_1 = mul = None
        convert_element_type: "bf16[512, 384]" = torch.ops.prims.convert_element_type.default(arg3_1, torch.bfloat16);  arg3_1 = None
        convert_element_type_1: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(mul_1, torch.bfloat16);  mul_1 = None
        view: "bf16[4096, 512]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_0);  convert_element_type_1 = _shape_param_0 = None
        mm: "bf16[4096, 384]" = torch.ops.aten.mm.default(view, convert_element_type);  view = convert_element_type = None
        view_2: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(mm, _shape_param_1);  mm = _shape_param_1 = None
        permute_1: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_2, [0, 2, 1, 3]);  view_2 = None
        clone: "bf16[32, 6, 128, 64]" = torch.ops.aten.clone.default(permute_1, memory_format = torch.contiguous_format);  permute_1 = None
        view_3: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone, _shape_param_2);  clone = _shape_param_2 = None
        permute_2: "bf16[192, 64, 128]" = torch.ops.aten.permute.default(view_3, [0, 2, 1])
        return (view_3, permute_2)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
