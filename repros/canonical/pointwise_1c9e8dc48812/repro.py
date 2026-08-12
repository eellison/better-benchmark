"""
Standalone repro captured via capture_hook.
Label: hf_GPTJForCausalLM_infer
Pattern hash: 1c9e8dc48812
Shape hash: 51251d0a
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 3
# Input shapes/strides/dtypes live in the sibling shapes.json (structured,
# one entry per point); forward()'s annotations document the default shapes
# inline. Default inputs = the first shapes.json point.

class Repro(torch.nn.Module):
    def forward(self, arg0_1: "bf16[2048, 64]", arg1_1: "i64[1, 128]", arg2_1: "bf16[128, 4096]", arg3_1: "bf16[128, 4096]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13, _shape_param_14, _shape_param_15, _shape_param_16, _shape_param_17, _shape_param_18):
        # No stacktrace found for following nodes
        repeat: "bf16[1, 2048, 64]" = torch.ops.aten.repeat.default(arg0_1, [1, 1, 1]);  arg0_1 = None
        unsqueeze: "i64[1, 128, 1]" = torch.ops.aten.unsqueeze.default(arg1_1, -1);  arg1_1 = None
        repeat_1: "i64[1, 128, 64]" = torch.ops.aten.repeat.default(unsqueeze, _shape_param_0);  unsqueeze = _shape_param_0 = None
        gather: "bf16[1, 128, 64]" = torch.ops.aten.gather.default(repeat, 1, repeat_1);  repeat = repeat_1 = None
        split = torch.ops.aten.split.Tensor(gather, 32, -1);  gather = None
        getitem: "bf16[1, 128, 32]" = split[0]
        getitem_1: "bf16[1, 128, 32]" = split[1];  split = None
        view: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(arg2_1, _shape_param_1);  arg2_1 = _shape_param_1 = None
        view_1: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view, _shape_param_2);  view = _shape_param_2 = None
        slice_1: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_1, 3, 0, 64)
        unsqueeze_1: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_1, 2)
        unsqueeze_2: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1, 4);  unsqueeze_1 = None
        expand: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_2, _shape_param_3);  unsqueeze_2 = _shape_param_3 = None
        clone: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand, memory_format = torch.contiguous_format);  expand = None
        view_2: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone, _shape_param_4);  clone = _shape_param_4 = None
        mul: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1, view_2);  view_2 = None
        slice_2: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_1, 3, 1, 9223372036854775807, 2)
        neg: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_2);  slice_2 = None
        unsqueeze_3: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg, 4);  neg = None
        slice_3: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_1, 3, 0, 9223372036854775807, 2);  slice_1 = None
        unsqueeze_4: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_3, 4);  slice_3 = None
        cat: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_3, unsqueeze_4], -1);  unsqueeze_3 = unsqueeze_4 = None
        view_3: "bf16[1, 128, 16, 64]" = torch.ops.aten.view.default(cat, _shape_param_5);  cat = _shape_param_5 = None
        unsqueeze_5: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem, 2)
        unsqueeze_6: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_5, 4);  unsqueeze_5 = None
        expand_1: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_6, _shape_param_6);  unsqueeze_6 = _shape_param_6 = None
        clone_1: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_1, memory_format = torch.contiguous_format);  expand_1 = None
        view_4: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_1, _shape_param_7);  clone_1 = _shape_param_7 = None
        mul_1: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_3, view_4);  view_3 = view_4 = None
        add: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul, mul_1);  mul = mul_1 = None
        slice_4: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_1, 3, 64, 9223372036854775807);  view_1 = None
        cat_1: "bf16[1, 128, 16, 256]" = torch.ops.aten.cat.default([add, slice_4], -1);  add = slice_4 = None
        permute: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_1, [0, 2, 1, 3]);  cat_1 = None
        convert_element_type: "f32[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute, torch.float32);  permute = None
        expand_2: "f32[1, 16, 128, 256]" = torch.ops.aten.expand.default(convert_element_type, _shape_param_8);  convert_element_type = _shape_param_8 = None
        view_5: "f32[16, 128, 256]" = torch.ops.aten.view.default(expand_2, _shape_param_9);  expand_2 = _shape_param_9 = None
        view_6: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(arg3_1, _shape_param_10);  arg3_1 = _shape_param_10 = None
        view_7: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_6, _shape_param_11);  view_6 = _shape_param_11 = None
        slice_5: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_7, 3, 0, 64)
        unsqueeze_7: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_1, 2);  getitem_1 = None
        unsqueeze_8: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_7, 4);  unsqueeze_7 = None
        expand_3: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_8, _shape_param_12);  unsqueeze_8 = _shape_param_12 = None
        clone_2: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_3, memory_format = torch.contiguous_format);  expand_3 = None
        view_8: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_2, _shape_param_13);  clone_2 = _shape_param_13 = None
        mul_2: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_5, view_8);  view_8 = None
        slice_6: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_5, 3, 1, 9223372036854775807, 2)
        neg_1: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_6);  slice_6 = None
        unsqueeze_9: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_1, 4);  neg_1 = None
        slice_7: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_5, 3, 0, 9223372036854775807, 2);  slice_5 = None
        unsqueeze_10: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_7, 4);  slice_7 = None
        cat_2: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_9, unsqueeze_10], -1);  unsqueeze_9 = unsqueeze_10 = None
        view_9: "bf16[1, 128, 16, 64]" = torch.ops.aten.view.default(cat_2, _shape_param_14);  cat_2 = _shape_param_14 = None
        unsqueeze_11: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem, 2);  getitem = None
        unsqueeze_12: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_11, 4);  unsqueeze_11 = None
        expand_4: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_12, _shape_param_15);  unsqueeze_12 = _shape_param_15 = None
        clone_3: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_4, memory_format = torch.contiguous_format);  expand_4 = None
        view_10: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_3, _shape_param_16);  clone_3 = _shape_param_16 = None
        mul_3: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_9, view_10);  view_9 = view_10 = None
        add_1: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_2, mul_3);  mul_2 = mul_3 = None
        slice_8: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_7, 3, 64, 9223372036854775807);  view_7 = None
        cat_3: "bf16[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_1, slice_8], -1);  add_1 = slice_8 = None
        permute_1: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_3, [0, 2, 1, 3]);  cat_3 = None
        convert_element_type_1: "f32[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_1, torch.float32);  permute_1 = None
        permute_2: "f32[1, 16, 256, 128]" = torch.ops.aten.permute.default(convert_element_type_1, [0, 1, 3, 2]);  convert_element_type_1 = None
        expand_5: "f32[1, 16, 256, 128]" = torch.ops.aten.expand.default(permute_2, _shape_param_17);  permute_2 = _shape_param_17 = None
        view_11: "f32[16, 256, 128]" = torch.ops.aten.view.default(expand_5, _shape_param_18);  expand_5 = _shape_param_18 = None
        return (view_5, view_11)



def _default_make_inputs():
    configs = load_shape_configs(__file__)
    if not configs:
        raise RuntimeError(
            "no shapes.json next to this repro — pass an explicit config "
            "via make_inputs(shape_config=...)")
    return make_inputs_from_config(next(iter(configs.values())))


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
