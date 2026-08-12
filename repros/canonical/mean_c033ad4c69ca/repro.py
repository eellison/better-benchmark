"""
Standalone repro captured via capture_hook.
Label: hf_Qwen/Qwen3-0.6B_infer
Pattern hash: c033ad4c69ca
Shape hash: da8b94aa
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
    def forward(self, arg0_1: "bf16[1000, 1024]", arg1_1: "bf16[1, 1000, 1024]", arg2_1: "bf16[1024]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # No stacktrace found for following nodes
        view: "bf16[1, 1000, 1024]" = torch.ops.aten.view.default(arg0_1, _shape_param_0);  arg0_1 = _shape_param_0 = None
        add: "bf16[1, 1000, 1024]" = torch.ops.aten.add.Tensor(arg1_1, view);  arg1_1 = view = None
        convert_element_type: "f32[1, 1000, 1024]" = torch.ops.prims.convert_element_type.default(add, torch.float32)
        pow_1: "f32[1, 1000, 1024]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type, 2)
        mean: "f32[1, 1000, 1]" = torch.ops.aten.mean.dim(pow_1, [-1], True);  pow_1 = None
        add_1: "f32[1, 1000, 1]" = torch.ops.aten.add.Tensor(mean, 1e-06);  mean = None
        rsqrt: "f32[1, 1000, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
        mul: "f32[1, 1000, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type, rsqrt);  convert_element_type = rsqrt = None
        convert_element_type_1: "bf16[1, 1000, 1024]" = torch.ops.prims.convert_element_type.default(mul, torch.bfloat16);  mul = None
        mul_1: "bf16[1, 1000, 1024]" = torch.ops.aten.mul.Tensor(arg2_1, convert_element_type_1);  arg2_1 = convert_element_type_1 = None
        view_1: "bf16[1000, 1024]" = torch.ops.aten.view.default(mul_1, _shape_param_1);  _shape_param_1 = None
        view_2: "bf16[1000, 1024]" = torch.ops.aten.view.default(mul_1, _shape_param_2);  _shape_param_2 = None
        view_3: "bf16[1000, 1024]" = torch.ops.aten.view.default(mul_1, _shape_param_3);  mul_1 = _shape_param_3 = None
        return (add, view_1, view_2, view_3)



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
