"""
Standalone repro captured via capture_hook.
Label: genai_rmsnorm_fwd_32768x256
Pattern hash: 86d0e8c4c31b
Shape hash: f98085da
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([32768, 256], bf16), T([256], f32))"

class Repro(torch.nn.Module):
    def forward(self, arg0_1: "bf16[32768, 256]", arg1_1: "f32[256]"):
        # File: /tmp/scratch_space/better_benchmark/capture_genai_kernels.py:269 in rmsnorm_fwd, code: x_f32 = x.float()
        convert_element_type_default: "f32[32768, 256]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.float32);  arg0_1 = None

        # File: /tmp/scratch_space/better_benchmark/capture_genai_kernels.py:272 in rmsnorm_fwd, code: * torch.rsqrt(torch.mean(x_f32.square(), dim=-1, keepdim=True) + 1e-6)
        pow_tensor_scalar: "f32[32768, 256]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_default, 2)
        mean_dim: "f32[32768, 1]" = torch.ops.aten.mean.dim(pow_tensor_scalar, [-1], True);  pow_tensor_scalar = None
        add_tensor: "f32[32768, 1]" = torch.ops.aten.add.Tensor(mean_dim, 1e-06);  mean_dim = None
        rsqrt_default: "f32[32768, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None

        # File: /tmp/scratch_space/better_benchmark/capture_genai_kernels.py:271 in rmsnorm_fwd, code: x_f32
        mul_tensor: "f32[32768, 256]" = torch.ops.aten.mul.Tensor(convert_element_type_default, rsqrt_default);  convert_element_type_default = rsqrt_default = None
        mul_tensor_1: "f32[32768, 256]" = torch.ops.aten.mul.Tensor(mul_tensor, arg1_1);  mul_tensor = arg1_1 = None

        # File: /tmp/scratch_space/better_benchmark/capture_genai_kernels.py:274 in rmsnorm_fwd, code: ).to(x.dtype)
        convert_element_type_default_1: "bf16[32768, 256]" = torch.ops.prims.convert_element_type.default(mul_tensor_1, torch.bfloat16);  mul_tensor_1 = None
        return convert_element_type_default_1


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
