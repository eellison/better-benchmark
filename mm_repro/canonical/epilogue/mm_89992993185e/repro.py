"""Curated MM repro. Pattern hash: 89992993185e."""
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
    def forward(self, arg0_1: "f32[4096, 4096]", arg1_1: "bf16[128, 4096]", arg2_1: "f32[2048, 64]", arg3_1: "i64[1, 128, 64]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7):
        # No stacktrace found for following nodes
        convert_element_type: "bf16[4096, 4096]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.bfloat16);  arg0_1 = None
        mm: "bf16[128, 4096]" = torch.ops.aten.mm.default(arg1_1, convert_element_type);  arg1_1 = convert_element_type = None
        view_1: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(mm, _shape_param_0);  mm = _shape_param_0 = None
        repeat: "f32[1, 2048, 64]" = torch.ops.aten.repeat.default(arg2_1, [1, 1, 1]);  arg2_1 = None
        gather: "f32[1, 128, 64]" = torch.ops.aten.gather.default(repeat, 1, arg3_1);  repeat = arg3_1 = None
        convert_element_type_1: "bf16[1, 128, 64]" = torch.ops.prims.convert_element_type.default(gather, torch.bfloat16);  gather = None
        split = torch.ops.aten.split.Tensor(convert_element_type_1, 32, -1);  convert_element_type_1 = None
        getitem: "bf16[1, 128, 32]" = split[0]
        getitem_1: "bf16[1, 128, 32]" = split[1];  split = None
        slice_1: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_1, 3, 0, 64)
        slice_2: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_1, 3, 64, 9223372036854775807);  view_1 = None
        unsqueeze: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem, 2);  getitem = None
        unsqueeze_1: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, 4);  unsqueeze = None
        expand: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_1, _shape_param_1);  unsqueeze_1 = _shape_param_1 = None
        clone: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand, memory_format = torch.contiguous_format);  expand = None
        view_2: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone, _shape_param_2);  clone = _shape_param_2 = None
        unsqueeze_2: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_1, 2);  getitem_1 = None
        unsqueeze_3: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, 4);  unsqueeze_2 = None
        expand_1: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_3, _shape_param_3);  unsqueeze_3 = _shape_param_3 = None
        clone_1: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_1, memory_format = torch.contiguous_format);  expand_1 = None
        view_3: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_1, _shape_param_4);  clone_1 = _shape_param_4 = None
        mul: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1, view_3);  view_3 = None
        slice_3: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_1, 3, 0, 9223372036854775807, 2)
        slice_4: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_1, 3, 1, 9223372036854775807, 2);  slice_1 = None
        neg: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_4);  slice_4 = None
        unsqueeze_4: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.view.default(neg, _shape_param_5);  neg = _shape_param_5 = None
        unsqueeze_5: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_3, 4);  slice_3 = None
        cat: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_4, unsqueeze_5], -1);  unsqueeze_4 = unsqueeze_5 = None
        view_4: "bf16[1, 128, 16, 64]" = torch.ops.aten.view.default(cat, _shape_param_6);  cat = _shape_param_6 = None
        mul_1: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_4, view_2);  view_4 = view_2 = None
        add: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul, mul_1);  mul = mul_1 = None
        cat_1: "bf16[1, 128, 16, 256]" = torch.ops.aten.cat.default([add, slice_2], -1);  add = slice_2 = None
        permute_1: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_1, [0, 2, 1, 3]);  cat_1 = None
        convert_element_type_2: "bf16[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_1, torch.bfloat16);  permute_1 = None
        view_5: "bf16[16, 128, 256]" = torch.ops.aten.view.default(convert_element_type_2, _shape_param_7);  convert_element_type_2 = _shape_param_7 = None
        permute_2: "bf16[16, 256, 128]" = torch.ops.aten.permute.default(view_5, [0, 2, 1])
        return (view_5, permute_2)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
