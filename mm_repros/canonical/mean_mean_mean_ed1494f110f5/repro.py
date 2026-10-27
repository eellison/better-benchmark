"""
Standalone repro captured via capture_hook.
Label: gemma-3-4b-it
Pattern hash: ed1494f110f5
Shape hash: ad0f589b
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
    def forward(self, arg0_1: "bf16[262208, 2560]", arg1_1: "i64[1, 1000]", arg2_1: "bf16[]", arg3_1: "bf16[2560]", arg4_1: "bf16[2048, 2560]", arg5_1: "bf16[256]", arg6_1: "bf16[128]", arg7_1: "bf16[1024, 2560]", arg8_1: "bf16[256]", arg9_1: "bf16[1024, 2560]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13, _shape_param_14, _shape_param_15, _shape_param_16):
        # No stacktrace found for following nodes
        embedding: "bf16[1, 1000, 2560]" = torch.ops.aten.embedding.default(arg0_1, arg1_1, 0);  arg0_1 = arg1_1 = None
        mul: "bf16[1, 1000, 2560]" = torch.ops.aten.mul.Tensor(embedding, arg2_1);  embedding = arg2_1 = None
        convert_element_type: "f32[1, 1000, 2560]" = torch.ops.prims.convert_element_type.default(mul, torch.float32)
        pow_1: "f32[1, 1000, 2560]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type, 2)
        mean: "f32[1, 1000, 1]" = torch.ops.aten.mean.dim(pow_1, [-1], True);  pow_1 = None
        add: "f32[1, 1000, 1]" = torch.ops.aten.add.Tensor(mean, 1e-06);  mean = None
        rsqrt: "f32[1, 1000, 1]" = torch.ops.aten.rsqrt.default(add);  add = None
        mul_1: "f32[1, 1000, 2560]" = torch.ops.aten.mul.Tensor(convert_element_type, rsqrt);  convert_element_type = rsqrt = None
        convert_element_type_1: "f32[2560]" = torch.ops.prims.convert_element_type.default(arg3_1, torch.float32);  arg3_1 = None
        add_1: "f32[2560]" = torch.ops.aten.add.Tensor(convert_element_type_1, 1.0);  convert_element_type_1 = None
        mul_2: "f32[1, 1000, 2560]" = torch.ops.aten.mul.Tensor(mul_1, add_1);  mul_1 = add_1 = None
        convert_element_type_2: "bf16[1, 1000, 2560]" = torch.ops.prims.convert_element_type.default(mul_2, torch.bfloat16);  mul_2 = None
        view: "bf16[1000, 2560]" = torch.ops.aten.view.default(convert_element_type_2, _shape_param_0);  _shape_param_0 = None
        permute: "bf16[2560, 2048]" = torch.ops.aten.permute.default(arg4_1, [1, 0]);  arg4_1 = None
        mm: "bf16[1000, 2048]" = torch.ops.aten.mm.default(view, permute);  view = permute = None
        view_1: "bf16[1, 1000, 2048]" = torch.ops.aten.view.default(mm, _shape_param_1);  mm = _shape_param_1 = None
        view_2: "bf16[1, 1000, 8, 256]" = torch.ops.aten.view.default(view_1, _shape_param_2);  view_1 = _shape_param_2 = None
        permute_1: "bf16[1, 8, 1000, 256]" = torch.ops.aten.permute.default(view_2, [0, 2, 1, 3]);  view_2 = None
        convert_element_type_3: "f32[1, 8, 1000, 256]" = torch.ops.prims.convert_element_type.default(permute_1, torch.float32);  permute_1 = None
        pow_2: "f32[1, 8, 1000, 256]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_3, 2)
        mean_1: "f32[1, 8, 1000, 1]" = torch.ops.aten.mean.dim(pow_2, [-1], True);  pow_2 = None
        add_2: "f32[1, 8, 1000, 1]" = torch.ops.aten.add.Tensor(mean_1, 1e-06);  mean_1 = None
        rsqrt_1: "f32[1, 8, 1000, 1]" = torch.ops.aten.rsqrt.default(add_2);  add_2 = None
        mul_3: "f32[1, 8, 1000, 256]" = torch.ops.aten.mul.Tensor(convert_element_type_3, rsqrt_1);  convert_element_type_3 = rsqrt_1 = None
        convert_element_type_4: "f32[256]" = torch.ops.prims.convert_element_type.default(arg5_1, torch.float32);  arg5_1 = None
        add_3: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_4, 1.0);  convert_element_type_4 = None
        mul_4: "f32[1, 8, 1000, 256]" = torch.ops.aten.mul.Tensor(mul_3, add_3);  mul_3 = add_3 = None
        convert_element_type_5: "bf16[1, 8, 1000, 256]" = torch.ops.prims.convert_element_type.default(mul_4, torch.bfloat16);  mul_4 = None
        unsqueeze: "bf16[1, 128]" = torch.ops.aten.unsqueeze.default(arg6_1, 0);  arg6_1 = None
        unsqueeze_1: "bf16[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, 2);  unsqueeze = None
        convert_element_type_6: "f32[1, 128, 1]" = torch.ops.prims.convert_element_type.default(unsqueeze_1, torch.float32);  unsqueeze_1 = None
        expand: "f32[1, 128, 1]" = torch.ops.aten.expand.default(convert_element_type_6, [1, -1, 1]);  convert_element_type_6 = None
        expand_1: "f32[1, 128, 1]" = torch.ops.aten.expand.default(expand, _shape_param_3);  expand = _shape_param_3 = None
        iota: "i64[1000]" = torch.ops.prims.iota.default(1000, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_4: "i64[1000]" = torch.ops.aten.add.Tensor(iota, 0);  iota = None
        unsqueeze_2: "i64[1, 1000]" = torch.ops.aten.unsqueeze.default(add_4, 0);  add_4 = None
        unsqueeze_3: "i64[1, 1, 1000]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, 1)
        convert_element_type_7: "f32[1, 1, 1000]" = torch.ops.prims.convert_element_type.default(unsqueeze_3, torch.float32);  unsqueeze_3 = None
        expand_2: "f32[1, 1, 1000]" = torch.ops.aten.expand.default(convert_element_type_7, _shape_param_4);  convert_element_type_7 = _shape_param_4 = None
        mul_5: "f32[1, 128, 1000]" = torch.ops.aten.mul.Tensor(expand_1, expand_2);  expand_1 = expand_2 = None
        permute_2: "f32[1, 1000, 128]" = torch.ops.aten.permute.default(mul_5, [0, 2, 1]);  mul_5 = None
        unsqueeze_4: "f32[1, 1000, 1, 128]" = torch.ops.aten.unsqueeze.default(permute_2, 2);  permute_2 = None
        expand_3: "f32[1, 1000, 2, 128]" = torch.ops.aten.expand.default(unsqueeze_4, _shape_param_5);  unsqueeze_4 = _shape_param_5 = None
        clone: "f32[1, 1000, 2, 128]" = torch.ops.aten.clone.default(expand_3, memory_format = torch.contiguous_format);  expand_3 = None
        view_3: "f32[1, 1000, 256]" = torch.ops.aten.view.default(clone, _shape_param_6);  clone = _shape_param_6 = None
        cos: "f32[1, 1000, 256]" = torch.ops.aten.cos.default(view_3)
        mul_6: "f32[1, 1000, 256]" = torch.ops.aten.mul.Tensor(cos, 1.0);  cos = None
        convert_element_type_8: "bf16[1, 1000, 256]" = torch.ops.prims.convert_element_type.default(mul_6, torch.bfloat16);  mul_6 = None
        unsqueeze_5: "bf16[1, 1, 1000, 256]" = torch.ops.aten.unsqueeze.default(convert_element_type_8, 1)
        mul_7: "bf16[1, 8, 1000, 256]" = torch.ops.aten.mul.Tensor(convert_element_type_5, unsqueeze_5)
        slice_1: "bf16[1, 8, 1000, 128]" = torch.ops.aten.slice.Tensor(convert_element_type_5, 3, 128, 9223372036854775807)
        neg: "bf16[1, 8, 1000, 128]" = torch.ops.aten.neg.default(slice_1);  slice_1 = None
        slice_2: "bf16[1, 8, 1000, 128]" = torch.ops.aten.slice.Tensor(convert_element_type_5, 3, 0, 128);  convert_element_type_5 = None
        cat: "bf16[1, 8, 1000, 256]" = torch.ops.aten.cat.default([neg, slice_2], -1);  neg = slice_2 = None
        sin: "f32[1, 1000, 256]" = torch.ops.aten.sin.default(view_3);  view_3 = None
        mul_8: "f32[1, 1000, 256]" = torch.ops.aten.mul.Tensor(sin, 1.0);  sin = None
        convert_element_type_9: "bf16[1, 1000, 256]" = torch.ops.prims.convert_element_type.default(mul_8, torch.bfloat16);  mul_8 = None
        unsqueeze_6: "bf16[1, 1, 1000, 256]" = torch.ops.aten.unsqueeze.default(convert_element_type_9, 1)
        mul_9: "bf16[1, 8, 1000, 256]" = torch.ops.aten.mul.Tensor(cat, unsqueeze_6);  cat = None
        add_5: "bf16[1, 8, 1000, 256]" = torch.ops.aten.add.Tensor(mul_7, mul_9);  mul_7 = mul_9 = None
        view_4: "bf16[1000, 2560]" = torch.ops.aten.view.default(convert_element_type_2, _shape_param_7);  _shape_param_7 = None
        permute_3: "bf16[2560, 1024]" = torch.ops.aten.permute.default(arg7_1, [1, 0]);  arg7_1 = None
        mm_1: "bf16[1000, 1024]" = torch.ops.aten.mm.default(view_4, permute_3);  view_4 = permute_3 = None
        view_5: "bf16[1, 1000, 1024]" = torch.ops.aten.view.default(mm_1, _shape_param_8);  mm_1 = _shape_param_8 = None
        view_6: "bf16[1, 1000, 4, 256]" = torch.ops.aten.view.default(view_5, _shape_param_9);  view_5 = _shape_param_9 = None
        permute_4: "bf16[1, 4, 1000, 256]" = torch.ops.aten.permute.default(view_6, [0, 2, 1, 3]);  view_6 = None
        convert_element_type_10: "f32[1, 4, 1000, 256]" = torch.ops.prims.convert_element_type.default(permute_4, torch.float32);  permute_4 = None
        pow_3: "f32[1, 4, 1000, 256]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_10, 2)
        mean_2: "f32[1, 4, 1000, 1]" = torch.ops.aten.mean.dim(pow_3, [-1], True);  pow_3 = None
        add_6: "f32[1, 4, 1000, 1]" = torch.ops.aten.add.Tensor(mean_2, 1e-06);  mean_2 = None
        rsqrt_2: "f32[1, 4, 1000, 1]" = torch.ops.aten.rsqrt.default(add_6);  add_6 = None
        mul_10: "f32[1, 4, 1000, 256]" = torch.ops.aten.mul.Tensor(convert_element_type_10, rsqrt_2);  convert_element_type_10 = rsqrt_2 = None
        convert_element_type_11: "f32[256]" = torch.ops.prims.convert_element_type.default(arg8_1, torch.float32);  arg8_1 = None
        add_7: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_11, 1.0);  convert_element_type_11 = None
        mul_11: "f32[1, 4, 1000, 256]" = torch.ops.aten.mul.Tensor(mul_10, add_7);  mul_10 = add_7 = None
        convert_element_type_12: "bf16[1, 4, 1000, 256]" = torch.ops.prims.convert_element_type.default(mul_11, torch.bfloat16);  mul_11 = None
        mul_12: "bf16[1, 4, 1000, 256]" = torch.ops.aten.mul.Tensor(convert_element_type_12, unsqueeze_5);  unsqueeze_5 = None
        slice_3: "bf16[1, 4, 1000, 128]" = torch.ops.aten.slice.Tensor(convert_element_type_12, 3, 128, 9223372036854775807)
        neg_1: "bf16[1, 4, 1000, 128]" = torch.ops.aten.neg.default(slice_3);  slice_3 = None
        slice_4: "bf16[1, 4, 1000, 128]" = torch.ops.aten.slice.Tensor(convert_element_type_12, 3, 0, 128);  convert_element_type_12 = None
        cat_1: "bf16[1, 4, 1000, 256]" = torch.ops.aten.cat.default([neg_1, slice_4], -1);  neg_1 = slice_4 = None
        mul_13: "bf16[1, 4, 1000, 256]" = torch.ops.aten.mul.Tensor(cat_1, unsqueeze_6);  cat_1 = unsqueeze_6 = None
        add_8: "bf16[1, 4, 1000, 256]" = torch.ops.aten.add.Tensor(mul_12, mul_13);  mul_12 = mul_13 = None
        unsqueeze_7: "bf16[1, 4, 1, 1000, 256]" = torch.ops.aten.unsqueeze.default(add_8, 2)
        expand_4: "bf16[1, 4, 2, 1000, 256]" = torch.ops.aten.expand.default(unsqueeze_7, _shape_param_10);  unsqueeze_7 = _shape_param_10 = None
        clone_1: "bf16[1, 4, 2, 1000, 256]" = torch.ops.aten.clone.default(expand_4, memory_format = torch.contiguous_format);  expand_4 = None
        view_7: "bf16[1, 8, 1000, 256]" = torch.ops.aten.view.default(clone_1, _shape_param_11);  clone_1 = _shape_param_11 = None
        view_8: "bf16[1000, 2560]" = torch.ops.aten.view.default(convert_element_type_2, _shape_param_12);  convert_element_type_2 = _shape_param_12 = None
        permute_5: "bf16[2560, 1024]" = torch.ops.aten.permute.default(arg9_1, [1, 0]);  arg9_1 = None
        mm_2: "bf16[1000, 1024]" = torch.ops.aten.mm.default(view_8, permute_5);  view_8 = permute_5 = None
        view_9: "bf16[1, 1000, 1024]" = torch.ops.aten.view.default(mm_2, _shape_param_13);  mm_2 = _shape_param_13 = None
        view_10: "bf16[1, 1000, 4, 256]" = torch.ops.aten.view.default(view_9, _shape_param_14);  view_9 = _shape_param_14 = None
        permute_6: "bf16[1, 4, 1000, 256]" = torch.ops.aten.permute.default(view_10, [0, 2, 1, 3]);  view_10 = None
        unsqueeze_8: "bf16[1, 4, 1, 1000, 256]" = torch.ops.aten.unsqueeze.default(permute_6, 2)
        expand_5: "bf16[1, 4, 2, 1000, 256]" = torch.ops.aten.expand.default(unsqueeze_8, _shape_param_15);  unsqueeze_8 = _shape_param_15 = None
        clone_2: "bf16[1, 4, 2, 1000, 256]" = torch.ops.aten.clone.default(expand_5, memory_format = torch.contiguous_format);  expand_5 = None
        view_11: "bf16[1, 8, 1000, 256]" = torch.ops.aten.view.default(clone_2, _shape_param_16);  clone_2 = _shape_param_16 = None
        slice_5: "bf16[1, 4, 1000, 256]" = torch.ops.aten.slice.Tensor(permute_6, 2, -1023, 9223372036854775807);  permute_6 = None
        slice_6: "bf16[1, 4, 1000, 256]" = torch.ops.aten.slice.Tensor(add_8, 2, -1023, 9223372036854775807);  add_8 = None
        return (mul, unsqueeze_2, convert_element_type_8, convert_element_type_9, add_5, view_7, view_11, slice_5, slice_6)



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
