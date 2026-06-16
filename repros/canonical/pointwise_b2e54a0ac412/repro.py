"""
Standalone repro captured via capture_hook.
Label: hf_GPTJForCausalLM_train
Pattern hash: b2e54a0ac412
Shape hash: 589b9793
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
    def forward(self, arg0_1: "bf16[128, 4096]", arg1_1: "bf16[128, 4096]", arg2_1: "f32[2048, 64]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13, _shape_param_14):
        # No stacktrace found for following nodes
        iota: "i64[128]" = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add: "i64[128]" = torch.ops.aten.add.Tensor(iota, 0);  iota = None
        unsqueeze: "i64[1, 128]" = torch.ops.aten.unsqueeze.default(add, 0);  add = None
        slice_1: "i64[1, 1]" = torch.ops.aten.slice.Tensor(unsqueeze, 1, 0, 1)
        sub: "i64[1, 1]" = torch.ops.aten.sub.Tensor(slice_1, 1);  slice_1 = None
        cat: "i64[1, 129]" = torch.ops.aten.cat.default([sub, unsqueeze], -1);  sub = None
        slice_2: "i64[1, 128]" = torch.ops.aten.slice.Tensor(cat, -1, 0, 128)
        slice_3: "i64[1, 128]" = torch.ops.aten.slice.Tensor(cat, -1, 1, 129);  cat = None
        sub_1: "i64[1, 128]" = torch.ops.aten.sub.Tensor(slice_3, slice_2);  slice_3 = slice_2 = None
        ne: "b8[1, 128]" = torch.ops.aten.ne.Scalar(sub_1, 1);  sub_1 = None
        view: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(arg0_1, _shape_param_0);  arg0_1 = _shape_param_0 = None
        view_1: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(arg1_1, _shape_param_1);  arg1_1 = _shape_param_1 = None
        view_2: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view, _shape_param_2);  view = _shape_param_2 = None
        view_3: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_1, _shape_param_3);  view_1 = _shape_param_3 = None
        repeat: "f32[1, 2048, 64]" = torch.ops.aten.repeat.default(arg2_1, [1, 1, 1]);  arg2_1 = None
        unsqueeze_1: "i64[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, -1)
        repeat_1: "i64[1, 128, 64]" = torch.ops.aten.repeat.default(unsqueeze_1, _shape_param_4);  unsqueeze_1 = _shape_param_4 = None
        gather: "f32[1, 128, 64]" = torch.ops.aten.gather.default(repeat, 1, repeat_1);  repeat = None
        convert_element_type: "bf16[1, 128, 64]" = torch.ops.prims.convert_element_type.default(gather, torch.bfloat16);  gather = None
        split = torch.ops.aten.split.Tensor(convert_element_type, 32, -1);  convert_element_type = None
        getitem: "bf16[1, 128, 32]" = split[0]
        getitem_1: "bf16[1, 128, 32]" = split[1];  split = None
        slice_4: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_3, 3, 0, 64)
        slice_5: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_3, 3, 64, 9223372036854775807);  view_3 = None
        slice_6: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_2, 3, 0, 64)
        slice_7: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_2, 3, 64, 9223372036854775807);  view_2 = None
        unsqueeze_2: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem, 2);  getitem = None
        unsqueeze_3: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, 4);  unsqueeze_2 = None
        expand: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_3, _shape_param_5);  _shape_param_5 = None
        clone: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand, memory_format = torch.contiguous_format);  expand = None
        view_4: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone, _shape_param_6);  clone = _shape_param_6 = None
        unsqueeze_4: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_1, 2);  getitem_1 = None
        unsqueeze_5: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, 4);  unsqueeze_4 = None
        expand_1: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_5, _shape_param_7);  _shape_param_7 = None
        clone_1: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_1, memory_format = torch.contiguous_format);  expand_1 = None
        view_5: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_1, _shape_param_8);  clone_1 = _shape_param_8 = None
        mul: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_4, view_5)
        slice_8: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_4, 3, 0, 9223372036854775807, 2)
        slice_9: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_4, 3, 1, 9223372036854775807, 2);  slice_4 = None
        neg: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_9);  slice_9 = None
        unsqueeze_6: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg, 4);  neg = None
        unsqueeze_7: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_8, 4);  slice_8 = None
        cat_1: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_6, unsqueeze_7], -1);  unsqueeze_6 = unsqueeze_7 = None
        view_6: "bf16[1, 128, 16, 64]" = torch.ops.aten.view.default(cat_1, _shape_param_9);  cat_1 = _shape_param_9 = None
        mul_1: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_6, view_4);  view_6 = None
        add_1: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul, mul_1);  mul = mul_1 = None
        mul_2: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_6, view_5);  view_5 = None
        slice_10: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_6, 3, 0, 9223372036854775807, 2)
        slice_11: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_6, 3, 1, 9223372036854775807, 2);  slice_6 = None
        neg_1: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_11);  slice_11 = None
        unsqueeze_8: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_1, 4);  neg_1 = None
        unsqueeze_9: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_10, 4);  slice_10 = None
        cat_2: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_8, unsqueeze_9], -1);  unsqueeze_8 = unsqueeze_9 = None
        view_7: "bf16[1, 128, 16, 64]" = torch.ops.aten.view.default(cat_2, _shape_param_10);  cat_2 = _shape_param_10 = None
        mul_3: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_7, view_4);  view_7 = view_4 = None
        add_2: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_2, mul_3);  mul_2 = mul_3 = None
        cat_3: "bf16[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_1, slice_5], -1);  add_1 = slice_5 = None
        cat_4: "bf16[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_2, slice_7], -1);  add_2 = slice_7 = None
        permute: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_3, [0, 2, 1, 3]);  cat_3 = None
        permute_1: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_4, [0, 2, 1, 3]);  cat_4 = None
        convert_element_type_1: "f32[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute, torch.float32);  permute = None
        permute_2: "f32[1, 16, 256, 128]" = torch.ops.aten.permute.default(convert_element_type_1, [0, 1, 3, 2]);  convert_element_type_1 = None
        convert_element_type_2: "bf16[1, 16, 256, 128]" = torch.ops.prims.convert_element_type.default(permute_2, torch.bfloat16);  permute_2 = None
        convert_element_type_3: "bf16[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_1, torch.bfloat16);  permute_1 = None
        expand_2: "bf16[1, 16, 128, 256]" = torch.ops.aten.expand.default(convert_element_type_3, _shape_param_11);  convert_element_type_3 = _shape_param_11 = None
        view_8: "bf16[16, 128, 256]" = torch.ops.aten.view.default(expand_2, _shape_param_12);  expand_2 = _shape_param_12 = None
        expand_3: "bf16[1, 16, 256, 128]" = torch.ops.aten.expand.default(convert_element_type_2, _shape_param_13);  convert_element_type_2 = _shape_param_13 = None
        view_9: "bf16[16, 256, 128]" = torch.ops.aten.view.default(expand_3, _shape_param_14);  expand_3 = _shape_param_14 = None
        permute_3: "bf16[16, 256, 128]" = torch.ops.aten.permute.default(view_8, [0, 2, 1])
        permute_4: "bf16[16, 128, 256]" = torch.ops.aten.permute.default(view_9, [0, 2, 1])
        return (unsqueeze, ne, repeat_1, unsqueeze_3, unsqueeze_5, view_8, view_9, permute_3, permute_4)



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
