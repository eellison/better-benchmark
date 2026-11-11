"""Curated MM repro. Pattern hash: 0c7c874497b4."""
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
    def forward(self, arg0_1: "f32[4096, 4096]", arg1_1: "bf16[128, 4096]", arg2_1: "f32[4096, 4096]", arg3_1: "f32[4096, 4096]", arg4_1: "bf16[1, 128, 1, 64]", arg5_1: "bf16[1, 128, 1, 64]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5):
        # No stacktrace found for following nodes
        convert_element_type: "bf16[4096, 4096]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.bfloat16);  arg0_1 = None
        mm: "bf16[128, 4096]" = torch.ops.aten.mm.default(arg1_1, convert_element_type);  convert_element_type = None
        convert_element_type_1: "bf16[4096, 4096]" = torch.ops.prims.convert_element_type.default(arg2_1, torch.bfloat16);  arg2_1 = None
        mm_1: "bf16[128, 4096]" = torch.ops.aten.mm.default(arg1_1, convert_element_type_1);  convert_element_type_1 = None
        convert_element_type_2: "bf16[4096, 4096]" = torch.ops.prims.convert_element_type.default(arg3_1, torch.bfloat16);  arg3_1 = None
        mm_2: "bf16[128, 4096]" = torch.ops.aten.mm.default(arg1_1, convert_element_type_2);  arg1_1 = convert_element_type_2 = None
        view_2: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(mm, _shape_param_0);  mm = _shape_param_0 = None
        view_3: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(mm_1, _shape_param_1);  mm_1 = _shape_param_1 = None
        slice_1: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_3, 3, 0, 64)
        slice_2: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_3, 3, 64, 9223372036854775807);  view_3 = None
        slice_3: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_2, 3, 0, 64)
        slice_4: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_2, 3, 64, 9223372036854775807);  view_2 = None
        mul: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1, arg4_1)
        slice_5: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_1, 3, 0, 9223372036854775807, 2)
        slice_6: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_1, 3, 1, 9223372036854775807, 2);  slice_1 = None
        neg: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_6);  slice_6 = None
        unsqueeze: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.view.default(neg, _shape_param_2);  neg = _shape_param_2 = None
        unsqueeze_1: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_5, 4);  slice_5 = None
        cat: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze, unsqueeze_1], -1);  unsqueeze = unsqueeze_1 = None
        view_4: "bf16[1, 128, 16, 64]" = torch.ops.aten.view.default(cat, _shape_param_3);  cat = _shape_param_3 = None
        mul_1: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_4, arg5_1);  view_4 = None
        add: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul, mul_1);  mul = mul_1 = None
        mul_2: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_3, arg4_1);  arg4_1 = None
        slice_7: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_3, 3, 0, 9223372036854775807, 2)
        slice_8: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_3, 3, 1, 9223372036854775807, 2);  slice_3 = None
        neg_1: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_8);  slice_8 = None
        unsqueeze_2: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.view.default(neg_1, _shape_param_4);  neg_1 = _shape_param_4 = None
        unsqueeze_3: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_7, 4);  slice_7 = None
        cat_1: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_2, unsqueeze_3], -1);  unsqueeze_2 = unsqueeze_3 = None
        view_5: "bf16[1, 128, 16, 64]" = torch.ops.aten.view.default(cat_1, _shape_param_5);  cat_1 = _shape_param_5 = None
        mul_3: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_5, arg5_1);  view_5 = arg5_1 = None
        add_1: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_2, mul_3);  mul_2 = mul_3 = None
        cat_2: "bf16[1, 128, 16, 256]" = torch.ops.aten.cat.default([add, slice_2], -1);  add = slice_2 = None
        cat_3: "bf16[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_1, slice_4], -1);  add_1 = slice_4 = None
        permute_3: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_3, [0, 2, 1, 3]);  cat_3 = None
        convert_element_type_3: "bf16[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_3, torch.bfloat16);  permute_3 = None
        return (mm_2, cat_2, convert_element_type_3)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
