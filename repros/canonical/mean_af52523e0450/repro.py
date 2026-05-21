"""
Standalone repro captured via capture_hook.
Label: vllm_Qwen_Qwen3-0.6B_000
Pattern hash: af52523e0450
Shape hash: cffba909
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([2048, 1024], bf16), T([4, 512, 1024], bf16), T([1024], bf16), S([4, 512, 1024]), S([2048, 1024]), S([2048, 1024]), S([2048, 1024]))"

class Repro(torch.nn.Module):
    def forward(self, mm_188: "bf16[2048, 1024]", add_242: "bf16[4, 512, 1024]", arg300_1: "bf16[1024]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # No stacktrace found for following nodes
        view_default: "bf16[4, 512, 1024]" = torch.ops.aten.view.default(mm_188, _shape_param_0);  mm_188 = _shape_param_0 = None
        add_tensor: "bf16[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_242, view_default);  add_242 = view_default = None
        convert_element_type_default: "f32[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_tensor, torch.float32);  add_tensor = None
        pow_tensor_scalar: "f32[4, 512, 1024]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_default, 2)
        mean_dim: "f32[4, 512, 1]" = torch.ops.aten.mean.dim(pow_tensor_scalar, [-1], True);  pow_tensor_scalar = None
        add_tensor_1: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(mean_dim, 1e-06);  mean_dim = None
        rsqrt_default: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        mul_tensor: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_default, rsqrt_default);  convert_element_type_default = rsqrt_default = None
        convert_element_type_default_1: "bf16[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(mul_tensor, torch.bfloat16);  mul_tensor = None
        mul_tensor_1: "bf16[4, 512, 1024]" = torch.ops.aten.mul.Tensor(arg300_1, convert_element_type_default_1);  arg300_1 = convert_element_type_default_1 = None
        view_default_1: "bf16[2048, 1024]" = torch.ops.aten.view.default(mul_tensor_1, _shape_param_1);  _shape_param_1 = None
        view_default_2: "bf16[2048, 1024]" = torch.ops.aten.view.default(mul_tensor_1, _shape_param_2);  _shape_param_2 = None
        view_default_3: "bf16[2048, 1024]" = torch.ops.aten.view.default(mul_tensor_1, _shape_param_3);  mul_tensor_1 = _shape_param_3 = None
        return (view_default_1, view_default_3, view_default_2)



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
