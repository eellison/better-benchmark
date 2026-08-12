"""
Standalone repro captured via capture_hook.
Label: hf_google/gemma-2-2b_infer
Pattern hash: 400b9e2117e6
Shape hash: 9e133e29
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
    def forward(self, arg0_1: "bf16[1000, 2048]", arg1_1: "bf16[1, 1000, 256]", arg2_1: "bf16[1, 1000, 256]", arg3_1: "bf16[1000, 1024]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5):
        # No stacktrace found for following nodes
        view: "bf16[1, 1000, 2048]" = torch.ops.aten.view.default(arg0_1, _shape_param_0);  arg0_1 = _shape_param_0 = None
        view_1: "bf16[1, 1000, 8, 256]" = torch.ops.aten.view.default(view, _shape_param_1);  view = _shape_param_1 = None
        permute: "bf16[1, 8, 1000, 256]" = torch.ops.aten.permute.default(view_1, [0, 2, 1, 3]);  view_1 = None
        unsqueeze: "bf16[1, 1, 1000, 256]" = torch.ops.aten.unsqueeze.default(arg1_1, 1);  arg1_1 = None
        mul: "bf16[1, 8, 1000, 256]" = torch.ops.aten.mul.Tensor(permute, unsqueeze)
        slice_1: "bf16[1, 8, 1000, 128]" = torch.ops.aten.slice.Tensor(permute, 3, 128, 9223372036854775807)
        neg: "bf16[1, 8, 1000, 128]" = torch.ops.aten.neg.default(slice_1);  slice_1 = None
        slice_2: "bf16[1, 8, 1000, 128]" = torch.ops.aten.slice.Tensor(permute, 3, 0, 128);  permute = None
        cat: "bf16[1, 8, 1000, 256]" = torch.ops.aten.cat.default([neg, slice_2], -1);  neg = slice_2 = None
        unsqueeze_1: "bf16[1, 1, 1000, 256]" = torch.ops.aten.unsqueeze.default(arg2_1, 1);  arg2_1 = None
        mul_1: "bf16[1, 8, 1000, 256]" = torch.ops.aten.mul.Tensor(cat, unsqueeze_1);  cat = None
        add: "bf16[1, 8, 1000, 256]" = torch.ops.aten.add.Tensor(mul, mul_1);  mul = mul_1 = None
        view_2: "bf16[1, 1000, 1024]" = torch.ops.aten.view.default(arg3_1, _shape_param_2);  arg3_1 = _shape_param_2 = None
        view_3: "bf16[1, 1000, 4, 256]" = torch.ops.aten.view.default(view_2, _shape_param_3);  view_2 = _shape_param_3 = None
        permute_1: "bf16[1, 4, 1000, 256]" = torch.ops.aten.permute.default(view_3, [0, 2, 1, 3]);  view_3 = None
        mul_2: "bf16[1, 4, 1000, 256]" = torch.ops.aten.mul.Tensor(permute_1, unsqueeze);  unsqueeze = None
        slice_3: "bf16[1, 4, 1000, 128]" = torch.ops.aten.slice.Tensor(permute_1, 3, 128, 9223372036854775807)
        neg_1: "bf16[1, 4, 1000, 128]" = torch.ops.aten.neg.default(slice_3);  slice_3 = None
        slice_4: "bf16[1, 4, 1000, 128]" = torch.ops.aten.slice.Tensor(permute_1, 3, 0, 128);  permute_1 = None
        cat_1: "bf16[1, 4, 1000, 256]" = torch.ops.aten.cat.default([neg_1, slice_4], -1);  neg_1 = slice_4 = None
        mul_3: "bf16[1, 4, 1000, 256]" = torch.ops.aten.mul.Tensor(cat_1, unsqueeze_1);  cat_1 = unsqueeze_1 = None
        add_1: "bf16[1, 4, 1000, 256]" = torch.ops.aten.add.Tensor(mul_2, mul_3);  mul_2 = mul_3 = None
        unsqueeze_2: "bf16[1, 4, 1, 1000, 256]" = torch.ops.aten.unsqueeze.default(add_1, 2)
        expand: "bf16[1, 4, 2, 1000, 256]" = torch.ops.aten.expand.default(unsqueeze_2, _shape_param_4);  unsqueeze_2 = _shape_param_4 = None
        clone: "bf16[1, 4, 2, 1000, 256]" = torch.ops.aten.clone.default(expand, memory_format = torch.contiguous_format);  expand = None
        view_4: "bf16[1, 8, 1000, 256]" = torch.ops.aten.view.default(clone, _shape_param_5);  clone = _shape_param_5 = None
        slice_5: "bf16[1, 4, 1000, 256]" = torch.ops.aten.slice.Tensor(add_1, 2, -4095, 9223372036854775807);  add_1 = None
        return (add, view_4, slice_5)



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
