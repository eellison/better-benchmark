"""Curated MM repro. Pattern hash: b0d2aa83947c."""
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
    def forward(self, arg0_1: "bf16[1024, 1024]", arg1_1: "bf16[1000, 1024]", arg2_1: "bf16[128]", arg3_1: "bf16[1, 1, 1000, 128]", arg4_1: "bf16[1, 1, 1000, 128]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        mm: "bf16[1000, 1024]" = torch.ops.aten.mm.default(arg1_1, arg0_1);  arg1_1 = arg0_1 = None
        view_1: "bf16[1, 1000, 8, 128]" = torch.ops.aten.view.default(mm, _shape_param_0);  mm = _shape_param_0 = None
        convert_element_type: "f32[1, 1000, 8, 128]" = torch.ops.prims.convert_element_type.default(view_1, torch.float32);  view_1 = None
        pow_1: "f32[1, 1000, 8, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type, 2)
        mean: "f32[1, 1000, 8, 1]" = torch.ops.aten.mean.dim(pow_1, [3], True);  pow_1 = None
        add: "f32[1, 1000, 8, 1]" = torch.ops.aten.add.Tensor(mean, 1e-06);  mean = None
        rsqrt: "f32[1, 1000, 8, 1]" = torch.ops.aten.rsqrt.default(add);  add = None
        mul: "f32[1, 1000, 8, 128]" = torch.ops.aten.mul.Tensor(convert_element_type, rsqrt);  convert_element_type = rsqrt = None
        convert_element_type_1: "bf16[1, 1000, 8, 128]" = torch.ops.prims.convert_element_type.default(mul, torch.bfloat16);  mul = None
        mul_1: "bf16[1, 1000, 8, 128]" = torch.ops.aten.mul.Tensor(arg2_1, convert_element_type_1);  arg2_1 = convert_element_type_1 = None
        permute_1: "bf16[1, 8, 1000, 128]" = torch.ops.aten.permute.default(mul_1, [0, 2, 1, 3]);  mul_1 = None
        mul_2: "bf16[1, 8, 1000, 128]" = torch.ops.aten.mul.Tensor(permute_1, arg3_1);  arg3_1 = None
        slice_1: "bf16[1, 8, 1000, 64]" = torch.ops.aten.slice.Tensor(permute_1, 3, 64, 9223372036854775807)
        neg: "bf16[1, 8, 1000, 64]" = torch.ops.aten.neg.default(slice_1);  slice_1 = None
        slice_2: "bf16[1, 8, 1000, 64]" = torch.ops.aten.slice.Tensor(permute_1, 3, 0, 64);  permute_1 = None
        cat: "bf16[1, 8, 1000, 128]" = torch.ops.aten.cat.default([neg, slice_2], -1);  neg = slice_2 = None
        mul_3: "bf16[1, 8, 1000, 128]" = torch.ops.aten.mul.Tensor(cat, arg4_1);  cat = arg4_1 = None
        add_1: "bf16[1, 8, 1000, 128]" = torch.ops.aten.add.Tensor(mul_2, mul_3);  mul_2 = mul_3 = None
        unsqueeze: "bf16[1, 8, 1, 1000, 128]" = torch.ops.aten.unsqueeze.default(add_1, 2)
        expand: "bf16[1, 8, 2, 1000, 128]" = torch.ops.aten.expand.default(unsqueeze, _shape_param_1);  unsqueeze = _shape_param_1 = None
        clone: "bf16[1, 8, 2, 1000, 128]" = torch.ops.aten.clone.default(expand, memory_format = torch.contiguous_format);  expand = None
        view_2: "bf16[1, 16, 1000, 128]" = torch.ops.aten.view.default(clone, _shape_param_2);  clone = _shape_param_2 = None
        return (add_1, view_2)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
