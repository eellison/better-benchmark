"""
Standalone repro captured via capture_hook.
Label: genai_patterns
Pattern hash: 6c5a53075222
Shape hash: 40db775a
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_prelude import *  # noqa: F401,F403
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, arg0_1: "bf16[4, 2048, 4096]", arg1_1: "bf16[4, 2048, 4096]", arg2_1: "bf16[4096]"):
        # File: /tmp/scratch_space/better_benchmark/capture_genai_patterns.py:168 in rmsnorm_residual_pattern, code: hidden_states = hidden_states + residual
        add_tensor: "bf16[4, 2048, 4096]" = torch.ops.aten.add.Tensor(arg0_1, arg1_1);  arg0_1 = arg1_1 = None

        # File: /tmp/scratch_space/better_benchmark/capture_genai_patterns.py:170 in rmsnorm_residual_pattern, code: variance = hidden_states.float().pow(2).mean(-1, keepdim=True)
        convert_element_type_default: "f32[4, 2048, 4096]" = torch.ops.prims.convert_element_type.default(add_tensor, torch.float32)
        pow_tensor_scalar: "f32[4, 2048, 4096]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_default, 2);  convert_element_type_default = None
        mean_dim: "f32[4, 2048, 1]" = torch.ops.aten.mean.dim(pow_tensor_scalar, [-1], True);  pow_tensor_scalar = None

        # File: /tmp/scratch_space/better_benchmark/capture_genai_patterns.py:171 in rmsnorm_residual_pattern, code: hidden_states = hidden_states * torch.rsqrt(variance + eps)
        add_tensor_1: "f32[4, 2048, 1]" = torch.ops.aten.add.Tensor(mean_dim, 1e-06);  mean_dim = None
        rsqrt_default: "f32[4, 2048, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        mul_tensor: "f32[4, 2048, 4096]" = torch.ops.aten.mul.Tensor(add_tensor, rsqrt_default);  add_tensor = rsqrt_default = None

        # File: /tmp/scratch_space/better_benchmark/capture_genai_patterns.py:172 in rmsnorm_residual_pattern, code: return (weight * hidden_states).to(hidden_states.dtype)
        mul_tensor_1: "f32[4, 2048, 4096]" = torch.ops.aten.mul.Tensor(arg2_1, mul_tensor);  arg2_1 = mul_tensor = None
        _output_to_half_0: "bf16[4, 2048, 4096]" = torch.ops.prims.convert_element_type.default(mul_tensor_1, torch.bfloat16);  mul_tensor_1 = None
        return _output_to_half_0


def _default_make_inputs():
    return [
    torch.randn([4, 2048, 4096], dtype=torch.bfloat16, device='cuda'),
    torch.randn([4, 2048, 4096], dtype=torch.bfloat16, device='cuda'),
    torch.randn([4096], dtype=torch.bfloat16, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
