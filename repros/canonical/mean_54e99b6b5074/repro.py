"""
Standalone repro captured via capture_hook.
Label: Gemma4_31B_text_6l_prefill_b1_s2048_sdpa_bf16
Pattern hash: 54e99b6b5074
Shape hash: 3fb7bf4d
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
    def forward(self, arg0_1: "bf16[2048, 4096]", arg1_1: "bf16[256]", arg2_1: "bf16[1, 2048, 256]", arg3_1: "bf16[1, 2048, 256]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # No stacktrace found for following nodes
        view: "bf16[1, 2048, 4096]" = torch.ops.aten.view.default(arg0_1, _shape_param_0);  arg0_1 = _shape_param_0 = None
        view_1: "bf16[1, 2048, 16, 256]" = torch.ops.aten.view.default(view, _shape_param_1);  view = _shape_param_1 = None
        convert_element_type: "f32[1, 2048, 16, 256]" = torch.ops.prims.convert_element_type.default(view_1, torch.float32);  view_1 = None
        pow_1: "f32[1, 2048, 16, 256]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type, 2)
        mean: "f32[1, 2048, 16, 1]" = torch.ops.aten.mean.dim(pow_1, [-1], True);  pow_1 = None
        add: "f32[1, 2048, 16, 1]" = torch.ops.aten.add.Tensor(mean, 1e-06);  mean = None
        pow_2: "f32[1, 2048, 16, 1]" = torch.ops.aten.pow.Tensor_Scalar(add, -0.5);  add = None
        mul: "f32[1, 2048, 16, 256]" = torch.ops.aten.mul.Tensor(convert_element_type, pow_2);  convert_element_type = pow_2 = None
        convert_element_type_1: "f32[256]" = torch.ops.prims.convert_element_type.default(arg1_1, torch.float32);  arg1_1 = None
        mul_1: "f32[1, 2048, 16, 256]" = torch.ops.aten.mul.Tensor(mul, convert_element_type_1);  mul = convert_element_type_1 = None
        convert_element_type_2: "bf16[1, 2048, 16, 256]" = torch.ops.prims.convert_element_type.default(mul_1, torch.bfloat16);  mul_1 = None
        unsqueeze: "bf16[1, 2048, 1, 256]" = torch.ops.aten.unsqueeze.default(arg2_1, 2);  arg2_1 = None
        mul_2: "bf16[1, 2048, 16, 256]" = torch.ops.aten.mul.Tensor(convert_element_type_2, unsqueeze);  unsqueeze = None
        slice_1: "bf16[1, 2048, 16, 128]" = torch.ops.aten.slice.Tensor(convert_element_type_2, 3, 128, 9223372036854775807)
        neg: "bf16[1, 2048, 16, 128]" = torch.ops.aten.neg.default(slice_1);  slice_1 = None
        slice_2: "bf16[1, 2048, 16, 128]" = torch.ops.aten.slice.Tensor(convert_element_type_2, 3, 0, 128);  convert_element_type_2 = None
        cat: "bf16[1, 2048, 16, 256]" = torch.ops.aten.cat.default([neg, slice_2], -1);  neg = slice_2 = None
        unsqueeze_1: "bf16[1, 2048, 1, 256]" = torch.ops.aten.unsqueeze.default(arg3_1, 2);  arg3_1 = None
        mul_3: "bf16[1, 2048, 16, 256]" = torch.ops.aten.mul.Tensor(cat, unsqueeze_1);  cat = unsqueeze_1 = None
        add_1: "bf16[1, 2048, 16, 256]" = torch.ops.aten.add.Tensor(mul_2, mul_3);  mul_2 = mul_3 = None
        permute: "bf16[1, 16, 2048, 256]" = torch.ops.aten.permute.default(add_1, [0, 2, 1, 3]);  add_1 = None
        unsqueeze_2: "bf16[1, 16, 1, 2048, 256]" = torch.ops.aten.unsqueeze.default(permute, 2);  permute = None
        expand: "bf16[1, 16, 2, 2048, 256]" = torch.ops.aten.expand.default(unsqueeze_2, _shape_param_2);  unsqueeze_2 = _shape_param_2 = None
        clone: "bf16[1, 16, 2, 2048, 256]" = torch.ops.aten.clone.default(expand, memory_format = torch.contiguous_format);  expand = None
        view_2: "bf16[1, 32, 2048, 256]" = torch.ops.aten.view.default(clone, _shape_param_3);  clone = _shape_param_3 = None
        return view_2



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
