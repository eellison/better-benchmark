"""
Standalone repro captured via capture_hook.
Label: Qwen3-0.6B
Pattern hash: 6f40f28895c4
Shape hash: 38c79a82
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
    def forward(self, arg0_1: "bf16[1, 16, 1000, 128]", arg1_1: "bf16[1024, 2048]", arg2_1: "bf16[1, 1000, 1024]", arg3_1: "bf16[1024]", arg4_1: "bf16[3072, 1024]", arg5_1: "bf16[3072, 1024]", arg6_1: "bf16[1024, 3072]", arg7_1: "bf16[1024]", arg8_1: "bf16[151936, 1024]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10):
        # No stacktrace found for following nodes
        permute: "bf16[1, 1000, 16, 128]" = torch.ops.aten.permute.default(arg0_1, [0, 2, 1, 3]);  arg0_1 = None
        view: "bf16[1, 1000, 2048]" = torch.ops.aten.view.default(permute, _shape_param_0);  permute = _shape_param_0 = None
        view_1: "bf16[1000, 2048]" = torch.ops.aten.view.default(view, _shape_param_1);  view = _shape_param_1 = None
        permute_1: "bf16[2048, 1024]" = torch.ops.aten.permute.default(arg1_1, [1, 0]);  arg1_1 = None
        mm: "bf16[1000, 1024]" = torch.ops.aten.mm.default(view_1, permute_1);  view_1 = permute_1 = None
        view_2: "bf16[1, 1000, 1024]" = torch.ops.aten.view.default(mm, _shape_param_2);  mm = _shape_param_2 = None
        add: "bf16[1, 1000, 1024]" = torch.ops.aten.add.Tensor(arg2_1, view_2);  arg2_1 = view_2 = None
        convert_element_type: "f32[1, 1000, 1024]" = torch.ops.prims.convert_element_type.default(add, torch.float32)
        pow_1: "f32[1, 1000, 1024]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type, 2)
        mean: "f32[1, 1000, 1]" = torch.ops.aten.mean.dim(pow_1, [-1], True);  pow_1 = None
        add_1: "f32[1, 1000, 1]" = torch.ops.aten.add.Tensor(mean, 1e-06);  mean = None
        rsqrt: "f32[1, 1000, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
        mul: "f32[1, 1000, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type, rsqrt);  convert_element_type = rsqrt = None
        convert_element_type_1: "bf16[1, 1000, 1024]" = torch.ops.prims.convert_element_type.default(mul, torch.bfloat16);  mul = None
        mul_1: "bf16[1, 1000, 1024]" = torch.ops.aten.mul.Tensor(arg3_1, convert_element_type_1);  arg3_1 = convert_element_type_1 = None
        view_3: "bf16[1000, 1024]" = torch.ops.aten.view.default(mul_1, _shape_param_3);  _shape_param_3 = None
        permute_2: "bf16[1024, 3072]" = torch.ops.aten.permute.default(arg4_1, [1, 0]);  arg4_1 = None
        mm_1: "bf16[1000, 3072]" = torch.ops.aten.mm.default(view_3, permute_2);  view_3 = permute_2 = None
        view_4: "bf16[1, 1000, 3072]" = torch.ops.aten.view.default(mm_1, _shape_param_4);  mm_1 = _shape_param_4 = None
        convert_element_type_2: "f32[1, 1000, 3072]" = torch.ops.prims.convert_element_type.default(view_4, torch.float32);  view_4 = None
        neg: "f32[1, 1000, 3072]" = torch.ops.aten.neg.default(convert_element_type_2)
        exp: "f32[1, 1000, 3072]" = torch.ops.aten.exp.default(neg);  neg = None
        add_2: "f32[1, 1000, 3072]" = torch.ops.aten.add.Tensor(exp, 1);  exp = None
        div: "f32[1, 1000, 3072]" = torch.ops.aten.div.Tensor(convert_element_type_2, add_2);  convert_element_type_2 = add_2 = None
        convert_element_type_3: "bf16[1, 1000, 3072]" = torch.ops.prims.convert_element_type.default(div, torch.bfloat16);  div = None
        view_5: "bf16[1000, 1024]" = torch.ops.aten.view.default(mul_1, _shape_param_5);  mul_1 = _shape_param_5 = None
        permute_3: "bf16[1024, 3072]" = torch.ops.aten.permute.default(arg5_1, [1, 0]);  arg5_1 = None
        mm_2: "bf16[1000, 3072]" = torch.ops.aten.mm.default(view_5, permute_3);  view_5 = permute_3 = None
        view_6: "bf16[1, 1000, 3072]" = torch.ops.aten.view.default(mm_2, _shape_param_6);  mm_2 = _shape_param_6 = None
        mul_2: "bf16[1, 1000, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_3, view_6);  convert_element_type_3 = view_6 = None
        view_7: "bf16[1000, 3072]" = torch.ops.aten.view.default(mul_2, _shape_param_7);  mul_2 = _shape_param_7 = None
        permute_4: "bf16[3072, 1024]" = torch.ops.aten.permute.default(arg6_1, [1, 0]);  arg6_1 = None
        mm_3: "bf16[1000, 1024]" = torch.ops.aten.mm.default(view_7, permute_4);  view_7 = permute_4 = None
        view_8: "bf16[1, 1000, 1024]" = torch.ops.aten.view.default(mm_3, _shape_param_8);  mm_3 = _shape_param_8 = None
        add_3: "bf16[1, 1000, 1024]" = torch.ops.aten.add.Tensor(add, view_8);  add = view_8 = None
        convert_element_type_4: "f32[1, 1000, 1024]" = torch.ops.prims.convert_element_type.default(add_3, torch.float32);  add_3 = None
        pow_2: "f32[1, 1000, 1024]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_4, 2)
        mean_1: "f32[1, 1000, 1]" = torch.ops.aten.mean.dim(pow_2, [-1], True);  pow_2 = None
        add_4: "f32[1, 1000, 1]" = torch.ops.aten.add.Tensor(mean_1, 1e-06);  mean_1 = None
        rsqrt_1: "f32[1, 1000, 1]" = torch.ops.aten.rsqrt.default(add_4);  add_4 = None
        mul_3: "f32[1, 1000, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_4, rsqrt_1);  convert_element_type_4 = rsqrt_1 = None
        convert_element_type_5: "bf16[1, 1000, 1024]" = torch.ops.prims.convert_element_type.default(mul_3, torch.bfloat16);  mul_3 = None
        mul_4: "bf16[1, 1000, 1024]" = torch.ops.aten.mul.Tensor(arg7_1, convert_element_type_5);  arg7_1 = convert_element_type_5 = None
        view_9: "bf16[1000, 1024]" = torch.ops.aten.view.default(mul_4, _shape_param_9);  mul_4 = _shape_param_9 = None
        permute_5: "bf16[1024, 151936]" = torch.ops.aten.permute.default(arg8_1, [1, 0]);  arg8_1 = None
        mm_4: "bf16[1000, 151936]" = torch.ops.aten.mm.default(view_9, permute_5);  view_9 = permute_5 = None
        view_10: "bf16[1, 1000, 151936]" = torch.ops.aten.view.default(mm_4, _shape_param_10);  mm_4 = _shape_param_10 = None
        return view_10



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
