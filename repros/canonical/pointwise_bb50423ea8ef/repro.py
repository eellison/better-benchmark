"""
Standalone repro captured via capture_hook.
Label: hf_GPTJForCausalLM_infer_000
Pattern hash: bb50423ea8ef
Shape hash: d46ce4c6
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([2048, 64], f32), T([1, 128], i64), T([128, 4096], f32), T([128, 4096], f32), S([1, 128, 4096]), S([1, 128, 16, 256]), S([1, 128, 1, 32, 2]), S([1, 128, 1, 64]), S([1, 128, 16, 64]), S([1, 128, 1, 32, 2]), S([1, 128, 1, 64]), S([1, 128, 4096]), S([1, 128, 16, 256]), S([1, 128, 1, 32, 2]), S([1, 128, 1, 64]), S([1, 128, 16, 64]), S([1, 128, 1, 32, 2]), S([1, 128, 1, 64]))"

class Repro(torch.nn.Module):
    def forward(self, arg304_1: "f32[2048, 64]", unsqueeze: "i64[1, 128]", mm_108: "f32[128, 4096]", mm_109: "f32[128, 4096]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13):
        # No stacktrace found for following nodes
        repeat_default: "f32[1, 2048, 64]" = torch.ops.aten.repeat.default(arg304_1, [1, 1, 1]);  arg304_1 = None
        unsqueeze_default: "i64[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
        repeat_default_1: "i64[1, 128, 64]" = torch.ops.aten.repeat.default(unsqueeze_default, [1, 1, 64]);  unsqueeze_default = None
        gather_default: "f32[1, 128, 64]" = torch.ops.aten.gather.default(repeat_default, 1, repeat_default_1);  repeat_default = repeat_default_1 = None
        split_tensor = torch.ops.aten.split.Tensor(gather_default, 32, -1);  gather_default = None
        getitem: "f32[1, 128, 32]" = split_tensor[0]
        getitem_1: "f32[1, 128, 32]" = split_tensor[1];  split_tensor = None
        view_default: "f32[1, 128, 4096]" = torch.ops.aten.view.default(mm_108, _shape_param_0);  mm_108 = _shape_param_0 = None
        view_default_1: "f32[1, 128, 16, 256]" = torch.ops.aten.view.default(view_default, _shape_param_1);  view_default = _shape_param_1 = None
        slice_tensor: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_default_1, 3, 0, 64)
        unsqueeze_default_1: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_1, 2)
        unsqueeze_default_2: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_1, 4);  unsqueeze_default_1 = None
        expand_default: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_default_2, _shape_param_2);  unsqueeze_default_2 = _shape_param_2 = None
        clone_default: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_default, memory_format = torch.contiguous_format);  expand_default = None
        view_default_2: "f32[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_default, _shape_param_3);  clone_default = _shape_param_3 = None
        mul_tensor: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_tensor, view_default_2);  view_default_2 = None
        slice_tensor_1: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_tensor, 3, 1, 9223372036854775807, 2)
        neg_default: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_tensor_1);  slice_tensor_1 = None
        unsqueeze_default_3: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_default, 4);  neg_default = None
        slice_tensor_2: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_tensor, 3, 0, 9223372036854775807, 2);  slice_tensor = None
        unsqueeze_default_4: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_tensor_2, 4);  slice_tensor_2 = None
        cat_default: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_default_3, unsqueeze_default_4], -1);  unsqueeze_default_3 = unsqueeze_default_4 = None
        view_default_3: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(cat_default, _shape_param_4);  cat_default = _shape_param_4 = None
        unsqueeze_default_5: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem, 2)
        unsqueeze_default_6: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_5, 4);  unsqueeze_default_5 = None
        expand_default_1: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_default_6, _shape_param_5);  unsqueeze_default_6 = _shape_param_5 = None
        clone_default_1: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_default_1, memory_format = torch.contiguous_format);  expand_default_1 = None
        view_default_4: "f32[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_default_1, _shape_param_6);  clone_default_1 = _shape_param_6 = None
        mul_tensor_1: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_default_3, view_default_4);  view_default_3 = view_default_4 = None
        add_tensor: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_tensor, mul_tensor_1);  mul_tensor = mul_tensor_1 = None
        slice_tensor_3: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_default_1, 3, 64, 9223372036854775807);  view_default_1 = None
        cat_default_1: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_tensor, slice_tensor_3], -1);  add_tensor = slice_tensor_3 = None
        permute_default: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_default_1, [0, 2, 1, 3]);  cat_default_1 = None
        view_default_5: "f32[1, 128, 4096]" = torch.ops.aten.view.default(mm_109, _shape_param_7);  mm_109 = _shape_param_7 = None
        view_default_6: "f32[1, 128, 16, 256]" = torch.ops.aten.view.default(view_default_5, _shape_param_8);  view_default_5 = _shape_param_8 = None
        slice_tensor_4: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_default_6, 3, 0, 64)
        unsqueeze_default_7: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_1, 2);  getitem_1 = None
        unsqueeze_default_8: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_7, 4);  unsqueeze_default_7 = None
        expand_default_2: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_default_8, _shape_param_9);  unsqueeze_default_8 = _shape_param_9 = None
        clone_default_2: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_default_2, memory_format = torch.contiguous_format);  expand_default_2 = None
        view_default_7: "f32[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_default_2, _shape_param_10);  clone_default_2 = _shape_param_10 = None
        mul_tensor_2: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_tensor_4, view_default_7);  view_default_7 = None
        slice_tensor_5: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_tensor_4, 3, 1, 9223372036854775807, 2)
        neg_default_1: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_tensor_5);  slice_tensor_5 = None
        unsqueeze_default_9: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_default_1, 4);  neg_default_1 = None
        slice_tensor_6: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_tensor_4, 3, 0, 9223372036854775807, 2);  slice_tensor_4 = None
        unsqueeze_default_10: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_tensor_6, 4);  slice_tensor_6 = None
        cat_default_2: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_default_9, unsqueeze_default_10], -1);  unsqueeze_default_9 = unsqueeze_default_10 = None
        view_default_8: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(cat_default_2, _shape_param_11);  cat_default_2 = _shape_param_11 = None
        unsqueeze_default_11: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem, 2);  getitem = None
        unsqueeze_default_12: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_11, 4);  unsqueeze_default_11 = None
        expand_default_3: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_default_12, _shape_param_12);  unsqueeze_default_12 = _shape_param_12 = None
        clone_default_3: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_default_3, memory_format = torch.contiguous_format);  expand_default_3 = None
        view_default_9: "f32[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_default_3, _shape_param_13);  clone_default_3 = _shape_param_13 = None
        mul_tensor_3: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_default_8, view_default_9);  view_default_8 = view_default_9 = None
        add_tensor_1: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_tensor_2, mul_tensor_3);  mul_tensor_2 = mul_tensor_3 = None
        slice_tensor_7: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_default_6, 3, 64, 9223372036854775807);  view_default_6 = None
        cat_default_3: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_tensor_1, slice_tensor_7], -1);  add_tensor_1 = slice_tensor_7 = None
        permute_default_1: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_default_3, [0, 2, 1, 3]);  cat_default_3 = None
        return (permute_default_1, permute_default)



def _default_make_inputs():
    from repro_harness import parse_shapes_config
    return parse_shapes_config(_shapes_config)


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
