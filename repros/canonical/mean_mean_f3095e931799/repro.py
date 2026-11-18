"""
Standalone repro captured via capture_hook.
Label: Gemma4_31B_text_6l_prefill_b1_s2048_sdpa_bf16
Pattern hash: f3095e931799
Shape hash: f6c36e9d
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
    def forward(self, arg0_1: "bf16[2048, 8192]", arg1_1: "bf16[256]", arg2_1: "f32[128]", arg3_1: "bf16[2048, 4096]", arg4_1: "bf16[256]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7):
        # No stacktrace found for following nodes
        view: "bf16[1, 2048, 8192]" = torch.ops.aten.view.default(arg0_1, _shape_param_0);  arg0_1 = _shape_param_0 = None
        view_1: "bf16[1, 2048, 32, 256]" = torch.ops.aten.view.default(view, _shape_param_1);  view = _shape_param_1 = None
        convert_element_type: "f32[1, 2048, 32, 256]" = torch.ops.prims.convert_element_type.default(view_1, torch.float32);  view_1 = None
        pow_1: "f32[1, 2048, 32, 256]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type, 2)
        mean: "f32[1, 2048, 32, 1]" = torch.ops.aten.mean.dim(pow_1, [-1], True);  pow_1 = None
        add: "f32[1, 2048, 32, 1]" = torch.ops.aten.add.Tensor(mean, 1e-06);  mean = None
        pow_2: "f32[1, 2048, 32, 1]" = torch.ops.aten.pow.Tensor_Scalar(add, -0.5);  add = None
        mul: "f32[1, 2048, 32, 256]" = torch.ops.aten.mul.Tensor(convert_element_type, pow_2);  convert_element_type = pow_2 = None
        convert_element_type_1: "f32[256]" = torch.ops.prims.convert_element_type.default(arg1_1, torch.float32);  arg1_1 = None
        mul_1: "f32[1, 2048, 32, 256]" = torch.ops.aten.mul.Tensor(mul, convert_element_type_1);  mul = convert_element_type_1 = None
        convert_element_type_2: "bf16[1, 2048, 32, 256]" = torch.ops.prims.convert_element_type.default(mul_1, torch.bfloat16);  mul_1 = None
        iota: "i64[2048]" = torch.ops.prims.iota.default(2048, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze: "i64[1, 2048]" = torch.ops.aten.unsqueeze.default(iota, 0);  iota = None
        unsqueeze_1: "i64[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, 2)
        convert_element_type_3: "f32[1, 2048, 1]" = torch.ops.prims.convert_element_type.default(unsqueeze_1, torch.float32);  unsqueeze_1 = None
        mul_2: "f32[1, 2048, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_3, arg2_1);  convert_element_type_3 = arg2_1 = None
        unsqueeze_2: "f32[1, 2048, 1, 128]" = torch.ops.aten.unsqueeze.default(mul_2, 2);  mul_2 = None
        expand: "f32[1, 2048, 2, 128]" = torch.ops.aten.expand.default(unsqueeze_2, _shape_param_2);  unsqueeze_2 = _shape_param_2 = None
        clone: "f32[1, 2048, 2, 128]" = torch.ops.aten.clone.default(expand, memory_format = torch.contiguous_format);  expand = None
        view_2: "f32[1, 2048, 256]" = torch.ops.aten.view.default(clone, _shape_param_3);  clone = _shape_param_3 = None
        cos: "f32[1, 2048, 256]" = torch.ops.aten.cos.default(view_2)
        convert_element_type_4: "bf16[1, 2048, 256]" = torch.ops.prims.convert_element_type.default(cos, torch.bfloat16);  cos = None
        unsqueeze_3: "bf16[1, 2048, 1, 256]" = torch.ops.aten.unsqueeze.default(convert_element_type_4, 2)
        mul_3: "bf16[1, 2048, 32, 256]" = torch.ops.aten.mul.Tensor(convert_element_type_2, unsqueeze_3);  unsqueeze_3 = None
        slice_1: "bf16[1, 2048, 32, 128]" = torch.ops.aten.slice.Tensor(convert_element_type_2, 3, 128, 9223372036854775807)
        neg: "bf16[1, 2048, 32, 128]" = torch.ops.aten.neg.default(slice_1);  slice_1 = None
        slice_2: "bf16[1, 2048, 32, 128]" = torch.ops.aten.slice.Tensor(convert_element_type_2, 3, 0, 128);  convert_element_type_2 = None
        cat: "bf16[1, 2048, 32, 256]" = torch.ops.aten.cat.default([neg, slice_2], -1);  neg = slice_2 = None
        sin: "f32[1, 2048, 256]" = torch.ops.aten.sin.default(view_2);  view_2 = None
        convert_element_type_5: "bf16[1, 2048, 256]" = torch.ops.prims.convert_element_type.default(sin, torch.bfloat16);  sin = None
        unsqueeze_4: "bf16[1, 2048, 1, 256]" = torch.ops.aten.unsqueeze.default(convert_element_type_5, 2)
        mul_4: "bf16[1, 2048, 32, 256]" = torch.ops.aten.mul.Tensor(cat, unsqueeze_4);  cat = unsqueeze_4 = None
        add_1: "bf16[1, 2048, 32, 256]" = torch.ops.aten.add.Tensor(mul_3, mul_4);  mul_3 = mul_4 = None
        permute: "bf16[1, 32, 2048, 256]" = torch.ops.aten.permute.default(add_1, [0, 2, 1, 3]);  add_1 = None
        view_3: "bf16[1, 2048, 4096]" = torch.ops.aten.view.default(arg3_1, _shape_param_4);  arg3_1 = _shape_param_4 = None
        view_4: "bf16[1, 2048, 16, 256]" = torch.ops.aten.view.default(view_3, _shape_param_5);  view_3 = _shape_param_5 = None
        convert_element_type_6: "f32[1, 2048, 16, 256]" = torch.ops.prims.convert_element_type.default(view_4, torch.float32);  view_4 = None
        pow_3: "f32[1, 2048, 16, 256]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_6, 2)
        mean_1: "f32[1, 2048, 16, 1]" = torch.ops.aten.mean.dim(pow_3, [-1], True);  pow_3 = None
        add_2: "f32[1, 2048, 16, 1]" = torch.ops.aten.add.Tensor(mean_1, 1e-06);  mean_1 = None
        pow_4: "f32[1, 2048, 16, 1]" = torch.ops.aten.pow.Tensor_Scalar(add_2, -0.5);  add_2 = None
        mul_5: "f32[1, 2048, 16, 256]" = torch.ops.aten.mul.Tensor(convert_element_type_6, pow_4);  convert_element_type_6 = pow_4 = None
        convert_element_type_7: "f32[256]" = torch.ops.prims.convert_element_type.default(arg4_1, torch.float32);  arg4_1 = None
        mul_6: "f32[1, 2048, 16, 256]" = torch.ops.aten.mul.Tensor(mul_5, convert_element_type_7);  mul_5 = convert_element_type_7 = None
        convert_element_type_8: "bf16[1, 2048, 16, 256]" = torch.ops.prims.convert_element_type.default(mul_6, torch.bfloat16);  mul_6 = None
        unsqueeze_5: "bf16[1, 2048, 1, 256]" = torch.ops.aten.unsqueeze.default(convert_element_type_4, 2)
        mul_7: "bf16[1, 2048, 16, 256]" = torch.ops.aten.mul.Tensor(convert_element_type_8, unsqueeze_5);  unsqueeze_5 = None
        slice_3: "bf16[1, 2048, 16, 128]" = torch.ops.aten.slice.Tensor(convert_element_type_8, 3, 128, 9223372036854775807)
        neg_1: "bf16[1, 2048, 16, 128]" = torch.ops.aten.neg.default(slice_3);  slice_3 = None
        slice_4: "bf16[1, 2048, 16, 128]" = torch.ops.aten.slice.Tensor(convert_element_type_8, 3, 0, 128);  convert_element_type_8 = None
        cat_1: "bf16[1, 2048, 16, 256]" = torch.ops.aten.cat.default([neg_1, slice_4], -1);  neg_1 = slice_4 = None
        unsqueeze_6: "bf16[1, 2048, 1, 256]" = torch.ops.aten.unsqueeze.default(convert_element_type_5, 2)
        mul_8: "bf16[1, 2048, 16, 256]" = torch.ops.aten.mul.Tensor(cat_1, unsqueeze_6);  cat_1 = unsqueeze_6 = None
        add_3: "bf16[1, 2048, 16, 256]" = torch.ops.aten.add.Tensor(mul_7, mul_8);  mul_7 = mul_8 = None
        permute_1: "bf16[1, 16, 2048, 256]" = torch.ops.aten.permute.default(add_3, [0, 2, 1, 3]);  add_3 = None
        unsqueeze_7: "bf16[1, 16, 1, 2048, 256]" = torch.ops.aten.unsqueeze.default(permute_1, 2);  permute_1 = None
        expand_1: "bf16[1, 16, 2, 2048, 256]" = torch.ops.aten.expand.default(unsqueeze_7, _shape_param_6);  unsqueeze_7 = _shape_param_6 = None
        clone_1: "bf16[1, 16, 2, 2048, 256]" = torch.ops.aten.clone.default(expand_1, memory_format = torch.contiguous_format);  expand_1 = None
        view_5: "bf16[1, 32, 2048, 256]" = torch.ops.aten.view.default(clone_1, _shape_param_7);  clone_1 = _shape_param_7 = None
        slice_5: "i64[1, 1]" = torch.ops.aten.slice.Tensor(unsqueeze, 1, 0, 1)
        sub: "i64[1, 1]" = torch.ops.aten.sub.Tensor(slice_5, 1);  slice_5 = None
        cat_2: "i64[1, 2049]" = torch.ops.aten.cat.default([sub, unsqueeze], -1);  sub = None
        slice_6: "i64[1, 2048]" = torch.ops.aten.slice.Tensor(cat_2, -1, 1, 2049)
        slice_7: "i64[1, 2048]" = torch.ops.aten.slice.Tensor(cat_2, -1, 0, 2048);  cat_2 = None
        sub_1: "i64[1, 2048]" = torch.ops.aten.sub.Tensor(slice_6, slice_7);  slice_6 = slice_7 = None
        ne: "b8[1, 2048]" = torch.ops.aten.ne.Scalar(sub_1, 1);  sub_1 = None
        slice_8: "i64[1, 1]" = torch.ops.aten.slice.Tensor(unsqueeze, 1, 0, 1)
        sub_2: "i64[1, 1]" = torch.ops.aten.sub.Tensor(slice_8, 1);  slice_8 = None
        cat_3: "i64[1, 2049]" = torch.ops.aten.cat.default([sub_2, unsqueeze], -1);  sub_2 = None
        slice_9: "i64[1, 2048]" = torch.ops.aten.slice.Tensor(cat_3, -1, 1, 2049)
        slice_10: "i64[1, 2048]" = torch.ops.aten.slice.Tensor(cat_3, -1, 0, 2048);  cat_3 = None
        sub_3: "i64[1, 2048]" = torch.ops.aten.sub.Tensor(slice_9, slice_10);  slice_9 = slice_10 = None
        ne_1: "b8[1, 2048]" = torch.ops.aten.ne.Scalar(sub_3, 1);  sub_3 = None
        return (unsqueeze, convert_element_type_4, convert_element_type_5, permute, view_5, ne, ne_1)



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
