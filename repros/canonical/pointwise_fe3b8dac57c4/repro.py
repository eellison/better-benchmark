"""
Standalone repro captured via capture_hook.
Label: hf_GPTJForCausalLM_train_000
Pattern hash: fe3b8dac57c4
Shape hash: 6612e5b3
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([128, 4096], f32), T([128, 4096], f32), T([2048, 64], f32), T([1, 128, 64], i64, gen=Index(2048)), S([1, 128, 4096]), S([1, 128, 4096]), S([1, 128, 16, 256]), S([1, 128, 16, 256]), S([1, 128, 1, 32, 2]), S([1, 128, 1, 64]), S([1, 128, 1, 32, 2]), S([1, 128, 1, 64]), S([1, 128, 16, 64]), S([1, 128, 16, 64]))"

class Repro(torch.nn.Module):
    def forward(self, mm_108: "f32[128, 4096]", mm_109: "f32[128, 4096]", arg304_1: "f32[2048, 64]", repeat_1: "i64[1, 128, 64]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9):
        # No stacktrace found for following nodes
        view_default: "f32[1, 128, 4096]" = torch.ops.aten.view.default(mm_108, _shape_param_0);  mm_108 = _shape_param_0 = None
        view_default_1: "f32[1, 128, 4096]" = torch.ops.aten.view.default(mm_109, _shape_param_1);  mm_109 = _shape_param_1 = None
        view_default_2: "f32[1, 128, 16, 256]" = torch.ops.aten.view.default(view_default, _shape_param_2);  view_default = _shape_param_2 = None
        view_default_3: "f32[1, 128, 16, 256]" = torch.ops.aten.view.default(view_default_1, _shape_param_3);  view_default_1 = _shape_param_3 = None
        repeat_default: "f32[1, 2048, 64]" = torch.ops.aten.repeat.default(arg304_1, [1, 1, 1]);  arg304_1 = None
        gather_default: "f32[1, 128, 64]" = torch.ops.aten.gather.default(repeat_default, 1, repeat_1);  repeat_default = repeat_1 = None
        split_tensor = torch.ops.aten.split.Tensor(gather_default, 32, -1);  gather_default = None
        getitem: "f32[1, 128, 32]" = split_tensor[0]
        getitem_1: "f32[1, 128, 32]" = split_tensor[1];  split_tensor = None
        slice_tensor: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_default_3, 3, 0, 64)
        slice_tensor_1: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_default_3, 3, 64, 9223372036854775807);  view_default_3 = None
        slice_tensor_2: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_default_2, 3, 0, 64)
        slice_tensor_3: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_default_2, 3, 64, 9223372036854775807);  view_default_2 = None
        unsqueeze_default: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem, 2);  getitem = None
        unsqueeze_default_1: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 4);  unsqueeze_default = None
        expand_default: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_default_1, _shape_param_4);  unsqueeze_default_1 = _shape_param_4 = None
        clone_default: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_default, memory_format = torch.contiguous_format);  expand_default = None
        view_default_4: "f32[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_default, _shape_param_5);  clone_default = _shape_param_5 = None
        unsqueeze_default_2: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_1, 2);  getitem_1 = None
        unsqueeze_default_3: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, 4);  unsqueeze_default_2 = None
        expand_default_1: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_default_3, _shape_param_6);  unsqueeze_default_3 = _shape_param_6 = None
        clone_default_1: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_default_1, memory_format = torch.contiguous_format);  expand_default_1 = None
        view_default_5: "f32[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_default_1, _shape_param_7);  clone_default_1 = _shape_param_7 = None
        mul_tensor: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_tensor, view_default_5)
        slice_tensor_4: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_tensor, 3, 0, 9223372036854775807, 2)
        slice_tensor_5: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_tensor, 3, 1, 9223372036854775807, 2);  slice_tensor = None
        neg_default: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_tensor_5);  slice_tensor_5 = None
        unsqueeze_default_4: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_default, 4);  neg_default = None
        unsqueeze_default_5: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_tensor_4, 4);  slice_tensor_4 = None
        cat_default: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_default_4, unsqueeze_default_5], -1);  unsqueeze_default_4 = unsqueeze_default_5 = None
        view_default_6: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(cat_default, _shape_param_8);  cat_default = _shape_param_8 = None
        mul_tensor_1: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_default_6, view_default_4);  view_default_6 = None
        add_tensor: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_tensor, mul_tensor_1);  mul_tensor = mul_tensor_1 = None
        mul_tensor_2: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_tensor_2, view_default_5);  view_default_5 = None
        slice_tensor_6: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_tensor_2, 3, 0, 9223372036854775807, 2)
        slice_tensor_7: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_tensor_2, 3, 1, 9223372036854775807, 2);  slice_tensor_2 = None
        neg_default_1: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_tensor_7);  slice_tensor_7 = None
        unsqueeze_default_6: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_default_1, 4);  neg_default_1 = None
        unsqueeze_default_7: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_tensor_6, 4);  slice_tensor_6 = None
        cat_default_1: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_default_6, unsqueeze_default_7], -1);  unsqueeze_default_6 = unsqueeze_default_7 = None
        view_default_7: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(cat_default_1, _shape_param_9);  cat_default_1 = _shape_param_9 = None
        mul_tensor_3: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_default_7, view_default_4);  view_default_7 = view_default_4 = None
        add_tensor_1: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_tensor_2, mul_tensor_3);  mul_tensor_2 = mul_tensor_3 = None
        cat_default_2: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_tensor, slice_tensor_1], -1);  add_tensor = slice_tensor_1 = None
        cat_default_3: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_tensor_1, slice_tensor_3], -1);  add_tensor_1 = slice_tensor_3 = None
        permute_default: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_default_2, [0, 2, 1, 3]);  cat_default_2 = None
        permute_default_1: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_default_3, [0, 2, 1, 3]);  cat_default_3 = None
        return (permute_default, permute_default_1)

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
