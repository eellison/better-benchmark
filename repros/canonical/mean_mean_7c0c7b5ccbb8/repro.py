"""
Standalone repro captured via capture_hook.
Label: hf_google/gemma-2-2b_infer
Pattern hash: 7c0c7b5ccbb8
Shape hash: ab924692
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
    def forward(self, arg0_1: "bf16[1000, 2304]", arg1_1: "bf16[2304]", arg2_1: "bf16[1, 1000, 2304]", arg3_1: "bf16[2304]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        view: "bf16[1, 1000, 2304]" = torch.ops.aten.view.default(arg0_1, _shape_param_0);  arg0_1 = _shape_param_0 = None
        convert_element_type: "f32[1, 1000, 2304]" = torch.ops.prims.convert_element_type.default(view, torch.float32);  view = None
        pow_1: "f32[1, 1000, 2304]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type, 2)
        mean: "f32[1, 1000, 1]" = torch.ops.aten.mean.dim(pow_1, [-1], True);  pow_1 = None
        add: "f32[1, 1000, 1]" = torch.ops.aten.add.Tensor(mean, 1e-06);  mean = None
        rsqrt: "f32[1, 1000, 1]" = torch.ops.aten.rsqrt.default(add);  add = None
        mul: "f32[1, 1000, 2304]" = torch.ops.aten.mul.Tensor(convert_element_type, rsqrt);  convert_element_type = rsqrt = None
        convert_element_type_1: "f32[2304]" = torch.ops.prims.convert_element_type.default(arg1_1, torch.float32);  arg1_1 = None
        add_1: "f32[2304]" = torch.ops.aten.add.Tensor(convert_element_type_1, 1.0);  convert_element_type_1 = None
        mul_1: "f32[1, 1000, 2304]" = torch.ops.aten.mul.Tensor(mul, add_1);  mul = add_1 = None
        convert_element_type_2: "bf16[1, 1000, 2304]" = torch.ops.prims.convert_element_type.default(mul_1, torch.bfloat16);  mul_1 = None
        add_2: "bf16[1, 1000, 2304]" = torch.ops.aten.add.Tensor(arg2_1, convert_element_type_2);  arg2_1 = convert_element_type_2 = None
        convert_element_type_3: "f32[1, 1000, 2304]" = torch.ops.prims.convert_element_type.default(add_2, torch.float32);  add_2 = None
        pow_2: "f32[1, 1000, 2304]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_3, 2)
        mean_1: "f32[1, 1000, 1]" = torch.ops.aten.mean.dim(pow_2, [-1], True);  pow_2 = None
        add_3: "f32[1, 1000, 1]" = torch.ops.aten.add.Tensor(mean_1, 1e-06);  mean_1 = None
        rsqrt_1: "f32[1, 1000, 1]" = torch.ops.aten.rsqrt.default(add_3);  add_3 = None
        mul_2: "f32[1, 1000, 2304]" = torch.ops.aten.mul.Tensor(convert_element_type_3, rsqrt_1);  convert_element_type_3 = rsqrt_1 = None
        convert_element_type_4: "f32[2304]" = torch.ops.prims.convert_element_type.default(arg3_1, torch.float32);  arg3_1 = None
        add_4: "f32[2304]" = torch.ops.aten.add.Tensor(convert_element_type_4, 1.0);  convert_element_type_4 = None
        mul_3: "f32[1, 1000, 2304]" = torch.ops.aten.mul.Tensor(mul_2, add_4);  mul_2 = add_4 = None
        convert_element_type_5: "bf16[1, 1000, 2304]" = torch.ops.prims.convert_element_type.default(mul_3, torch.bfloat16);  mul_3 = None
        view_1: "bf16[1000, 2304]" = torch.ops.aten.view.default(convert_element_type_5, _shape_param_1);  convert_element_type_5 = _shape_param_1 = None
        return view_1



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
