"""Curated MM repro. Pattern hash: 9ca544a99039."""
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
    def forward(self, arg0_1: "bf16[4096, 4096]", arg1_1: "bf16[128, 4096]", arg2_1: "bf16[1, 128, 1, 64]", arg3_1: "bf16[1, 128, 1, 64]", arg4_1: "bf16[4096, 4096]", arg5_1: "bf16[128, 4096]", arg6_1: "bf16[1, 128, 1, 64]", arg7_1: "bf16[1, 128, 1, 64]", arg8_1: "bf16[4096, 4096]", arg9_1: "bf16[128, 4096]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8):
        # No stacktrace found for following nodes
        mm: "bf16[128, 4096]" = torch.ops.aten.mm.default(arg1_1, arg0_1);  arg1_1 = arg0_1 = None
        view_1: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(mm, _shape_param_0);  mm = _shape_param_0 = None
        slice_1: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_1, 3, 0, 64)
        mul: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1, arg2_1);  arg2_1 = None
        slice_2: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_1, 3, 1, 9223372036854775807, 2)
        neg: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_2);  slice_2 = None
        unsqueeze: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.view.default(neg, _shape_param_1);  neg = _shape_param_1 = None
        slice_3: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_1, 3, 0, 9223372036854775807, 2);  slice_1 = None
        unsqueeze_1: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_3, 4);  slice_3 = None
        cat: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze, unsqueeze_1], -1);  unsqueeze = unsqueeze_1 = None
        view_2: "bf16[1, 128, 16, 64]" = torch.ops.aten.view.default(cat, _shape_param_2);  cat = _shape_param_2 = None
        mul_1: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_2, arg3_1);  view_2 = arg3_1 = None
        add: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul, mul_1);  mul = mul_1 = None
        slice_4: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_1, 3, 64, 9223372036854775807);  view_1 = None
        cat_1: "bf16[1, 128, 16, 256]" = torch.ops.aten.cat.default([add, slice_4], -1);  add = slice_4 = None
        permute_1: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_1, [0, 2, 1, 3]);  cat_1 = None
        convert_element_type: "f32[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_1, torch.float32);  permute_1 = None
        view_3: "f32[16, 128, 256]" = torch.ops.aten.view.default(convert_element_type, _shape_param_3);  convert_element_type = _shape_param_3 = None
        mm_1: "bf16[128, 4096]" = torch.ops.aten.mm.default(arg5_1, arg4_1);  arg5_1 = arg4_1 = None
        view_5: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(mm_1, _shape_param_4);  mm_1 = _shape_param_4 = None
        slice_5: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_5, 3, 0, 64)
        mul_2: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_5, arg6_1);  arg6_1 = None
        slice_6: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_5, 3, 1, 9223372036854775807, 2)
        neg_1: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_6);  slice_6 = None
        unsqueeze_2: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.view.default(neg_1, _shape_param_5);  neg_1 = _shape_param_5 = None
        slice_7: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_5, 3, 0, 9223372036854775807, 2);  slice_5 = None
        unsqueeze_3: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_7, 4);  slice_7 = None
        cat_2: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_2, unsqueeze_3], -1);  unsqueeze_2 = unsqueeze_3 = None
        view_6: "bf16[1, 128, 16, 64]" = torch.ops.aten.view.default(cat_2, _shape_param_6);  cat_2 = _shape_param_6 = None
        mul_3: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_6, arg7_1);  view_6 = arg7_1 = None
        add_1: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_2, mul_3);  mul_2 = mul_3 = None
        slice_8: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_5, 3, 64, 9223372036854775807);  view_5 = None
        cat_3: "bf16[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_1, slice_8], -1);  add_1 = slice_8 = None
        permute_3: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_3, [0, 2, 1, 3]);  cat_3 = None
        convert_element_type_1: "f32[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_3, torch.float32);  permute_3 = None
        permute_4: "f32[1, 16, 256, 128]" = torch.ops.aten.permute.default(convert_element_type_1, [0, 1, 3, 2]);  convert_element_type_1 = None
        mm_2: "bf16[128, 4096]" = torch.ops.aten.mm.default(arg9_1, arg8_1);  arg9_1 = arg8_1 = None
        view_8: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(mm_2, _shape_param_7);  mm_2 = _shape_param_7 = None
        permute_6: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_8, [0, 2, 1, 3]);  view_8 = None
        view_9: "bf16[16, 128, 256]" = torch.ops.aten.view.default(permute_6, _shape_param_8);  permute_6 = _shape_param_8 = None
        return (view_3, permute_4, view_9)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
