"""
Standalone repro captured via capture_hook.
Label: tritonbench_welford_ln_262144x1024
Pattern hash: 6f459d40ee4d
Shape hash: 507054c1
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
    def forward(self, arg0_1: "bf16[262144, 1024]", arg1_1: "bf16[1024]", arg2_1: "bf16[1024]"):
        # File: /tmp/scratch_space/better_benchmark/capture_tritonbench_ops.py:275 in welford_layer_norm, code: x_fp32 = x.to(torch.float32)
        convert_element_type_default: "f32[262144, 1024]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.float32);  arg0_1 = None

        # File: /tmp/scratch_space/better_benchmark/capture_tritonbench_ops.py:276 in welford_layer_norm, code: mean = x_fp32.mean(dim=-1, keepdim=True)
        mean_dim: "f32[262144, 1]" = torch.ops.aten.mean.dim(convert_element_type_default, [-1], True)

        # File: /tmp/scratch_space/better_benchmark/capture_tritonbench_ops.py:278 in welford_layer_norm, code: x_hat = (x_fp32 - mean) * torch.rsqrt(var + eps)
        sub_tensor: "f32[262144, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_default, mean_dim)

        # File: /tmp/scratch_space/better_benchmark/capture_tritonbench_ops.py:277 in welford_layer_norm, code: var = ((x_fp32 - mean) ** 2).mean(dim=-1, keepdim=True)
        sub_tensor_1: "f32[262144, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_default, mean_dim);  convert_element_type_default = mean_dim = None
        pow_tensor_scalar: "f32[262144, 1024]" = torch.ops.aten.pow.Tensor_Scalar(sub_tensor_1, 2);  sub_tensor_1 = None
        mean_dim_1: "f32[262144, 1]" = torch.ops.aten.mean.dim(pow_tensor_scalar, [-1], True);  pow_tensor_scalar = None

        # File: /tmp/scratch_space/better_benchmark/capture_tritonbench_ops.py:278 in welford_layer_norm, code: x_hat = (x_fp32 - mean) * torch.rsqrt(var + eps)
        add_tensor: "f32[262144, 1]" = torch.ops.aten.add.Tensor(mean_dim_1, 1e-05);  mean_dim_1 = None
        rsqrt_default: "f32[262144, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        mul_tensor: "f32[262144, 1024]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None

        # File: /tmp/scratch_space/better_benchmark/capture_tritonbench_ops.py:279 in welford_layer_norm, code: return (weight * x_hat + bias).to(x.dtype)
        mul_tensor_1: "f32[262144, 1024]" = torch.ops.aten.mul.Tensor(arg1_1, mul_tensor);  arg1_1 = mul_tensor = None
        add_tensor_1: "f32[262144, 1024]" = torch.ops.aten.add.Tensor(mul_tensor_1, arg2_1);  mul_tensor_1 = arg2_1 = None
        convert_element_type_default_1: "bf16[262144, 1024]" = torch.ops.prims.convert_element_type.default(add_tensor_1, torch.bfloat16);  add_tensor_1 = None
        return convert_element_type_default_1


def _default_make_inputs():
    return [
    torch.randn([262144, 1024], dtype=torch.bfloat16, device='cuda'),
    torch.randn([1024], dtype=torch.bfloat16, device='cuda'),
    torch.randn([1024], dtype=torch.bfloat16, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
