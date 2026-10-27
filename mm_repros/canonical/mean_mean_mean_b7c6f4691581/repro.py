"""
Standalone repro captured via capture_hook.
Label: gemma-2-2b
Pattern hash: b7c6f4691581
Shape hash: c47dec30
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
    def forward(self, arg0_1: "bf16[1, 8, 1000, 256]", arg1_1: "bf16[2304, 2048]", arg2_1: "bf16[2304]", arg3_1: "bf16[1, 1000, 2304]", arg4_1: "bf16[2304]", arg5_1: "bf16[9216, 2304]", arg6_1: "bf16[9216, 2304]", arg7_1: "bf16[2304, 9216]", arg8_1: "bf16[2304]", arg9_1: "bf16[2304]", arg10_1: "bf16[256000, 2304]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10):
        # No stacktrace found for following nodes
        permute: "bf16[1, 1000, 8, 256]" = torch.ops.aten.permute.default(arg0_1, [0, 2, 1, 3]);  arg0_1 = None
        view: "bf16[1, 1000, 2048]" = torch.ops.aten.view.default(permute, _shape_param_0);  permute = _shape_param_0 = None
        view_1: "bf16[1000, 2048]" = torch.ops.aten.view.default(view, _shape_param_1);  view = _shape_param_1 = None
        permute_1: "bf16[2048, 2304]" = torch.ops.aten.permute.default(arg1_1, [1, 0]);  arg1_1 = None
        mm: "bf16[1000, 2304]" = torch.ops.aten.mm.default(view_1, permute_1);  view_1 = permute_1 = None
        view_2: "bf16[1, 1000, 2304]" = torch.ops.aten.view.default(mm, _shape_param_2);  mm = _shape_param_2 = None
        convert_element_type: "f32[1, 1000, 2304]" = torch.ops.prims.convert_element_type.default(view_2, torch.float32);  view_2 = None
        pow_1: "f32[1, 1000, 2304]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type, 2)
        mean: "f32[1, 1000, 1]" = torch.ops.aten.mean.dim(pow_1, [-1], True);  pow_1 = None
        add: "f32[1, 1000, 1]" = torch.ops.aten.add.Tensor(mean, 1e-06);  mean = None
        rsqrt: "f32[1, 1000, 1]" = torch.ops.aten.rsqrt.default(add);  add = None
        mul: "f32[1, 1000, 2304]" = torch.ops.aten.mul.Tensor(convert_element_type, rsqrt);  convert_element_type = rsqrt = None
        convert_element_type_1: "f32[2304]" = torch.ops.prims.convert_element_type.default(arg2_1, torch.float32);  arg2_1 = None
        add_1: "f32[2304]" = torch.ops.aten.add.Tensor(convert_element_type_1, 1.0);  convert_element_type_1 = None
        mul_1: "f32[1, 1000, 2304]" = torch.ops.aten.mul.Tensor(mul, add_1);  mul = add_1 = None
        convert_element_type_2: "bf16[1, 1000, 2304]" = torch.ops.prims.convert_element_type.default(mul_1, torch.bfloat16);  mul_1 = None
        add_2: "bf16[1, 1000, 2304]" = torch.ops.aten.add.Tensor(arg3_1, convert_element_type_2);  arg3_1 = convert_element_type_2 = None
        convert_element_type_3: "f32[1, 1000, 2304]" = torch.ops.prims.convert_element_type.default(add_2, torch.float32)
        pow_2: "f32[1, 1000, 2304]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_3, 2)
        mean_1: "f32[1, 1000, 1]" = torch.ops.aten.mean.dim(pow_2, [-1], True);  pow_2 = None
        add_3: "f32[1, 1000, 1]" = torch.ops.aten.add.Tensor(mean_1, 1e-06);  mean_1 = None
        rsqrt_1: "f32[1, 1000, 1]" = torch.ops.aten.rsqrt.default(add_3);  add_3 = None
        mul_2: "f32[1, 1000, 2304]" = torch.ops.aten.mul.Tensor(convert_element_type_3, rsqrt_1);  convert_element_type_3 = rsqrt_1 = None
        convert_element_type_4: "f32[2304]" = torch.ops.prims.convert_element_type.default(arg4_1, torch.float32);  arg4_1 = None
        add_4: "f32[2304]" = torch.ops.aten.add.Tensor(convert_element_type_4, 1.0);  convert_element_type_4 = None
        mul_3: "f32[1, 1000, 2304]" = torch.ops.aten.mul.Tensor(mul_2, add_4);  mul_2 = add_4 = None
        convert_element_type_5: "bf16[1, 1000, 2304]" = torch.ops.prims.convert_element_type.default(mul_3, torch.bfloat16);  mul_3 = None
        view_3: "bf16[1000, 2304]" = torch.ops.aten.view.default(convert_element_type_5, _shape_param_3);  _shape_param_3 = None
        permute_2: "bf16[2304, 9216]" = torch.ops.aten.permute.default(arg5_1, [1, 0]);  arg5_1 = None
        mm_1: "bf16[1000, 9216]" = torch.ops.aten.mm.default(view_3, permute_2);  view_3 = permute_2 = None
        view_4: "bf16[1, 1000, 9216]" = torch.ops.aten.view.default(mm_1, _shape_param_4);  mm_1 = _shape_param_4 = None
        convert_element_type_6: "f32[1, 1000, 9216]" = torch.ops.prims.convert_element_type.default(view_4, torch.float32);  view_4 = None
        mul_4: "f32[1, 1000, 9216]" = torch.ops.aten.mul.Tensor(convert_element_type_6, 0.5)
        mul_5: "f32[1, 1000, 9216]" = torch.ops.aten.mul.Tensor(convert_element_type_6, convert_element_type_6)
        mul_6: "f32[1, 1000, 9216]" = torch.ops.aten.mul.Tensor(mul_5, convert_element_type_6);  mul_5 = None
        mul_7: "f32[1, 1000, 9216]" = torch.ops.aten.mul.Tensor(mul_6, 0.044715);  mul_6 = None
        add_5: "f32[1, 1000, 9216]" = torch.ops.aten.add.Tensor(convert_element_type_6, mul_7);  convert_element_type_6 = mul_7 = None
        mul_8: "f32[1, 1000, 9216]" = torch.ops.aten.mul.Tensor(add_5, 0.7978845608028654);  add_5 = None
        tanh: "f32[1, 1000, 9216]" = torch.ops.aten.tanh.default(mul_8);  mul_8 = None
        add_6: "f32[1, 1000, 9216]" = torch.ops.aten.add.Tensor(tanh, 1);  tanh = None
        mul_9: "f32[1, 1000, 9216]" = torch.ops.aten.mul.Tensor(mul_4, add_6);  mul_4 = add_6 = None
        convert_element_type_7: "bf16[1, 1000, 9216]" = torch.ops.prims.convert_element_type.default(mul_9, torch.bfloat16);  mul_9 = None
        view_5: "bf16[1000, 2304]" = torch.ops.aten.view.default(convert_element_type_5, _shape_param_5);  convert_element_type_5 = _shape_param_5 = None
        permute_3: "bf16[2304, 9216]" = torch.ops.aten.permute.default(arg6_1, [1, 0]);  arg6_1 = None
        mm_2: "bf16[1000, 9216]" = torch.ops.aten.mm.default(view_5, permute_3);  view_5 = permute_3 = None
        view_6: "bf16[1, 1000, 9216]" = torch.ops.aten.view.default(mm_2, _shape_param_6);  mm_2 = _shape_param_6 = None
        mul_10: "bf16[1, 1000, 9216]" = torch.ops.aten.mul.Tensor(convert_element_type_7, view_6);  convert_element_type_7 = view_6 = None
        view_7: "bf16[1000, 9216]" = torch.ops.aten.view.default(mul_10, _shape_param_7);  mul_10 = _shape_param_7 = None
        permute_4: "bf16[9216, 2304]" = torch.ops.aten.permute.default(arg7_1, [1, 0]);  arg7_1 = None
        mm_3: "bf16[1000, 2304]" = torch.ops.aten.mm.default(view_7, permute_4);  view_7 = permute_4 = None
        view_8: "bf16[1, 1000, 2304]" = torch.ops.aten.view.default(mm_3, _shape_param_8);  mm_3 = _shape_param_8 = None
        convert_element_type_8: "f32[1, 1000, 2304]" = torch.ops.prims.convert_element_type.default(view_8, torch.float32);  view_8 = None
        pow_3: "f32[1, 1000, 2304]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_8, 2)
        mean_2: "f32[1, 1000, 1]" = torch.ops.aten.mean.dim(pow_3, [-1], True);  pow_3 = None
        add_7: "f32[1, 1000, 1]" = torch.ops.aten.add.Tensor(mean_2, 1e-06);  mean_2 = None
        rsqrt_2: "f32[1, 1000, 1]" = torch.ops.aten.rsqrt.default(add_7);  add_7 = None
        mul_11: "f32[1, 1000, 2304]" = torch.ops.aten.mul.Tensor(convert_element_type_8, rsqrt_2);  convert_element_type_8 = rsqrt_2 = None
        convert_element_type_9: "f32[2304]" = torch.ops.prims.convert_element_type.default(arg8_1, torch.float32);  arg8_1 = None
        add_8: "f32[2304]" = torch.ops.aten.add.Tensor(convert_element_type_9, 1.0);  convert_element_type_9 = None
        mul_12: "f32[1, 1000, 2304]" = torch.ops.aten.mul.Tensor(mul_11, add_8);  mul_11 = add_8 = None
        convert_element_type_10: "bf16[1, 1000, 2304]" = torch.ops.prims.convert_element_type.default(mul_12, torch.bfloat16);  mul_12 = None
        add_9: "bf16[1, 1000, 2304]" = torch.ops.aten.add.Tensor(add_2, convert_element_type_10);  add_2 = convert_element_type_10 = None
        convert_element_type_11: "f32[1, 1000, 2304]" = torch.ops.prims.convert_element_type.default(add_9, torch.float32);  add_9 = None
        pow_4: "f32[1, 1000, 2304]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_11, 2)
        mean_3: "f32[1, 1000, 1]" = torch.ops.aten.mean.dim(pow_4, [-1], True);  pow_4 = None
        add_10: "f32[1, 1000, 1]" = torch.ops.aten.add.Tensor(mean_3, 1e-06);  mean_3 = None
        rsqrt_3: "f32[1, 1000, 1]" = torch.ops.aten.rsqrt.default(add_10);  add_10 = None
        mul_13: "f32[1, 1000, 2304]" = torch.ops.aten.mul.Tensor(convert_element_type_11, rsqrt_3);  convert_element_type_11 = rsqrt_3 = None
        convert_element_type_12: "f32[2304]" = torch.ops.prims.convert_element_type.default(arg9_1, torch.float32);  arg9_1 = None
        add_11: "f32[2304]" = torch.ops.aten.add.Tensor(convert_element_type_12, 1.0);  convert_element_type_12 = None
        mul_14: "f32[1, 1000, 2304]" = torch.ops.aten.mul.Tensor(mul_13, add_11);  mul_13 = add_11 = None
        convert_element_type_13: "bf16[1, 1000, 2304]" = torch.ops.prims.convert_element_type.default(mul_14, torch.bfloat16);  mul_14 = None
        view_9: "bf16[1000, 2304]" = torch.ops.aten.view.default(convert_element_type_13, _shape_param_9);  convert_element_type_13 = _shape_param_9 = None
        permute_5: "bf16[2304, 256000]" = torch.ops.aten.permute.default(arg10_1, [1, 0]);  arg10_1 = None
        mm_4: "bf16[1000, 256000]" = torch.ops.aten.mm.default(view_9, permute_5);  view_9 = permute_5 = None
        view_10: "bf16[1, 1000, 256000]" = torch.ops.aten.view.default(mm_4, _shape_param_10);  mm_4 = _shape_param_10 = None
        div: "bf16[1, 1000, 256000]" = torch.ops.aten.div.Tensor(view_10, 30.0);  view_10 = None
        tanh_1: "bf16[1, 1000, 256000]" = torch.ops.aten.tanh.default(div);  div = None
        mul_15: "bf16[1, 1000, 256000]" = torch.ops.aten.mul.Tensor(tanh_1, 30.0);  tanh_1 = None
        return mul_15



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
