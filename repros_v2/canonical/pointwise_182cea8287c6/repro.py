"""
Standalone repro captured via capture_hook.
Label: hf_google/gemma-2-2b_infer
Pattern hash: 182cea8287c6
Shape hash: 5da3d9cf
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
    def forward(self, arg0_1: "bf16[1000, 2048]", arg1_1: "bf16[128]", arg2_1: "bf16[1000, 1024]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9):
        # No stacktrace found for following nodes
        view: "bf16[1, 1000, 2048]" = torch.ops.aten.view.default(arg0_1, _shape_param_0);  arg0_1 = _shape_param_0 = None
        view_1: "bf16[1, 1000, 8, 256]" = torch.ops.aten.view.default(view, _shape_param_1);  view = _shape_param_1 = None
        permute: "bf16[1, 8, 1000, 256]" = torch.ops.aten.permute.default(view_1, [0, 2, 1, 3]);  view_1 = None
        unsqueeze: "bf16[1, 128]" = torch.ops.aten.unsqueeze.default(arg1_1, 0);  arg1_1 = None
        unsqueeze_1: "bf16[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, 2);  unsqueeze = None
        convert_element_type: "f32[1, 128, 1]" = torch.ops.prims.convert_element_type.default(unsqueeze_1, torch.float32);  unsqueeze_1 = None
        expand: "f32[1, 128, 1]" = torch.ops.aten.expand.default(convert_element_type, [1, -1, 1]);  convert_element_type = None
        expand_1: "f32[1, 128, 1]" = torch.ops.aten.expand.default(expand, _shape_param_2);  expand = _shape_param_2 = None
        iota: "i64[1000]" = torch.ops.prims.iota.default(1000, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add: "i64[1000]" = torch.ops.aten.add.Tensor(iota, 0);  iota = None
        unsqueeze_2: "i64[1, 1000]" = torch.ops.aten.unsqueeze.default(add, 0);  add = None
        unsqueeze_3: "i64[1, 1, 1000]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, 1);  unsqueeze_2 = None
        convert_element_type_1: "f32[1, 1, 1000]" = torch.ops.prims.convert_element_type.default(unsqueeze_3, torch.float32);  unsqueeze_3 = None
        expand_2: "f32[1, 1, 1000]" = torch.ops.aten.expand.default(convert_element_type_1, _shape_param_3);  convert_element_type_1 = _shape_param_3 = None
        mul: "f32[1, 128, 1000]" = torch.ops.aten.mul.Tensor(expand_1, expand_2);  expand_1 = expand_2 = None
        permute_1: "f32[1, 1000, 128]" = torch.ops.aten.permute.default(mul, [0, 2, 1]);  mul = None
        unsqueeze_4: "f32[1, 1000, 1, 128]" = torch.ops.aten.unsqueeze.default(permute_1, 2);  permute_1 = None
        expand_3: "f32[1, 1000, 2, 128]" = torch.ops.aten.expand.default(unsqueeze_4, _shape_param_4);  unsqueeze_4 = _shape_param_4 = None
        clone: "f32[1, 1000, 2, 128]" = torch.ops.aten.clone.default(expand_3, memory_format = torch.contiguous_format);  expand_3 = None
        view_2: "f32[1, 1000, 256]" = torch.ops.aten.view.default(clone, _shape_param_5);  clone = _shape_param_5 = None
        cos: "f32[1, 1000, 256]" = torch.ops.aten.cos.default(view_2)
        mul_1: "f32[1, 1000, 256]" = torch.ops.aten.mul.Tensor(cos, 1.0);  cos = None
        convert_element_type_2: "bf16[1, 1000, 256]" = torch.ops.prims.convert_element_type.default(mul_1, torch.bfloat16);  mul_1 = None
        unsqueeze_5: "bf16[1, 1, 1000, 256]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1)
        mul_2: "bf16[1, 8, 1000, 256]" = torch.ops.aten.mul.Tensor(permute, unsqueeze_5)
        slice_1: "bf16[1, 8, 1000, 128]" = torch.ops.aten.slice.Tensor(permute, 3, 128, 9223372036854775807)
        neg: "bf16[1, 8, 1000, 128]" = torch.ops.aten.neg.default(slice_1);  slice_1 = None
        slice_2: "bf16[1, 8, 1000, 128]" = torch.ops.aten.slice.Tensor(permute, 3, 0, 128);  permute = None
        cat: "bf16[1, 8, 1000, 256]" = torch.ops.aten.cat.default([neg, slice_2], -1);  neg = slice_2 = None
        sin: "f32[1, 1000, 256]" = torch.ops.aten.sin.default(view_2);  view_2 = None
        mul_3: "f32[1, 1000, 256]" = torch.ops.aten.mul.Tensor(sin, 1.0);  sin = None
        convert_element_type_3: "bf16[1, 1000, 256]" = torch.ops.prims.convert_element_type.default(mul_3, torch.bfloat16);  mul_3 = None
        unsqueeze_6: "bf16[1, 1, 1000, 256]" = torch.ops.aten.unsqueeze.default(convert_element_type_3, 1)
        mul_4: "bf16[1, 8, 1000, 256]" = torch.ops.aten.mul.Tensor(cat, unsqueeze_6);  cat = None
        add_1: "bf16[1, 8, 1000, 256]" = torch.ops.aten.add.Tensor(mul_2, mul_4);  mul_2 = mul_4 = None
        view_3: "bf16[1, 1000, 1024]" = torch.ops.aten.view.default(arg2_1, _shape_param_6);  arg2_1 = _shape_param_6 = None
        view_4: "bf16[1, 1000, 4, 256]" = torch.ops.aten.view.default(view_3, _shape_param_7);  view_3 = _shape_param_7 = None
        permute_2: "bf16[1, 4, 1000, 256]" = torch.ops.aten.permute.default(view_4, [0, 2, 1, 3]);  view_4 = None
        mul_5: "bf16[1, 4, 1000, 256]" = torch.ops.aten.mul.Tensor(permute_2, unsqueeze_5);  unsqueeze_5 = None
        slice_3: "bf16[1, 4, 1000, 128]" = torch.ops.aten.slice.Tensor(permute_2, 3, 128, 9223372036854775807)
        neg_1: "bf16[1, 4, 1000, 128]" = torch.ops.aten.neg.default(slice_3);  slice_3 = None
        slice_4: "bf16[1, 4, 1000, 128]" = torch.ops.aten.slice.Tensor(permute_2, 3, 0, 128);  permute_2 = None
        cat_1: "bf16[1, 4, 1000, 256]" = torch.ops.aten.cat.default([neg_1, slice_4], -1);  neg_1 = slice_4 = None
        mul_6: "bf16[1, 4, 1000, 256]" = torch.ops.aten.mul.Tensor(cat_1, unsqueeze_6);  cat_1 = unsqueeze_6 = None
        add_2: "bf16[1, 4, 1000, 256]" = torch.ops.aten.add.Tensor(mul_5, mul_6);  mul_5 = mul_6 = None
        unsqueeze_7: "bf16[1, 4, 1, 1000, 256]" = torch.ops.aten.unsqueeze.default(add_2, 2)
        expand_4: "bf16[1, 4, 2, 1000, 256]" = torch.ops.aten.expand.default(unsqueeze_7, _shape_param_8);  unsqueeze_7 = _shape_param_8 = None
        clone_1: "bf16[1, 4, 2, 1000, 256]" = torch.ops.aten.clone.default(expand_4, memory_format = torch.contiguous_format);  expand_4 = None
        view_5: "bf16[1, 8, 1000, 256]" = torch.ops.aten.view.default(clone_1, _shape_param_9);  clone_1 = _shape_param_9 = None
        slice_5: "bf16[1, 4, 1000, 256]" = torch.ops.aten.slice.Tensor(add_2, 2, -4095, 9223372036854775807);  add_2 = None
        return (convert_element_type_2, convert_element_type_3, add_1, view_5, slice_5)



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
