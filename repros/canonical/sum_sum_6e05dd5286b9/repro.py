"""
Standalone repro captured via capture_hook.
Label: genai_rmsnorm_bwd_32768x256
Pattern hash: 6e05dd5286b9
Shape hash: c4f6330b
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
    def forward(self, tangents_1: "bf16[]", _shape_param_0, primals_1: "bf16[32768, 256]", rsqrt: "f32[32768, 1]", primals_2: "f32[256]", _shape_param_1, _shape_param_2):
        # File: /tmp/scratch_space/better_benchmark/capture_genai_kernels.py:306 in rmsnorm_bwd, code: return out.sum()
        expand_default: "bf16[32768, 256]" = torch.ops.aten.expand.default(tangents_1, _shape_param_0);  tangents_1 = _shape_param_0 = None

        # File: /tmp/scratch_space/better_benchmark/capture_genai_kernels.py:305 in rmsnorm_bwd, code: ).to(x.dtype)
        convert_element_type_default: "f32[32768, 256]" = torch.ops.prims.convert_element_type.default(expand_default, torch.float32);  expand_default = None

        # File: /tmp/scratch_space/better_benchmark/capture_genai_kernels.py:300 in rmsnorm_bwd, code: x_f32 = x.float()
        convert_element_type_default_1: "f32[32768, 256]" = torch.ops.prims.convert_element_type.default(primals_1, torch.float32);  primals_1 = None

        # File: /tmp/scratch_space/better_benchmark/capture_genai_kernels.py:302 in rmsnorm_bwd, code: x_f32
        mul_tensor: "f32[32768, 256]" = torch.ops.aten.mul.Tensor(convert_element_type_default_1, rsqrt)
        mul_tensor_1: "f32[32768, 256]" = torch.ops.aten.mul.Tensor(convert_element_type_default, mul_tensor);  mul_tensor = None
        mul_tensor_2: "f32[32768, 256]" = torch.ops.aten.mul.Tensor(convert_element_type_default, primals_2);  convert_element_type_default = primals_2 = None
        sum_dim_int_list: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(mul_tensor_1, [0], True);  mul_tensor_1 = None
        reshape_default: "f32[256]" = torch.ops.aten.reshape.default(sum_dim_int_list, _shape_param_1);  sum_dim_int_list = _shape_param_1 = None
        mul_tensor_3: "f32[32768, 256]" = torch.ops.aten.mul.Tensor(mul_tensor_2, convert_element_type_default_1)
        mul_tensor_4: "f32[32768, 256]" = torch.ops.aten.mul.Tensor(mul_tensor_2, rsqrt);  mul_tensor_2 = None
        sum_dim_int_list_1: "f32[32768, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_3, [1], True);  mul_tensor_3 = None

        # File: /tmp/scratch_space/better_benchmark/capture_genai_kernels.py:303 in rmsnorm_bwd, code: * torch.rsqrt(torch.mean(x_f32.square(), dim=-1, keepdim=True) + 1e-6)
        pow_tensor_scalar: "f32[32768, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt, 3);  rsqrt = None
        mul_scalar: "f32[32768, 1]" = torch.ops.aten.mul.Scalar(sum_dim_int_list_1, -0.5);  sum_dim_int_list_1 = None
        mul_tensor_5: "f32[32768, 1]" = torch.ops.aten.mul.Tensor(mul_scalar, pow_tensor_scalar);  mul_scalar = pow_tensor_scalar = None
        expand_default_1: "f32[32768, 256]" = torch.ops.aten.expand.default(mul_tensor_5, _shape_param_2);  mul_tensor_5 = _shape_param_2 = None
        div_scalar: "f32[32768, 256]" = torch.ops.aten.div.Scalar(expand_default_1, 256);  expand_default_1 = None
        pow_tensor_scalar_1: "f32[32768, 256]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_default_1, 1.0);  convert_element_type_default_1 = None
        mul_scalar_1: "f32[32768, 256]" = torch.ops.aten.mul.Scalar(pow_tensor_scalar_1, 2.0);  pow_tensor_scalar_1 = None
        mul_tensor_6: "f32[32768, 256]" = torch.ops.aten.mul.Tensor(div_scalar, mul_scalar_1);  div_scalar = mul_scalar_1 = None
        add_tensor: "f32[32768, 256]" = torch.ops.aten.add.Tensor(mul_tensor_4, mul_tensor_6);  mul_tensor_4 = mul_tensor_6 = None

        # File: /tmp/scratch_space/better_benchmark/capture_genai_kernels.py:300 in rmsnorm_bwd, code: x_f32 = x.float()
        convert_element_type_default_2: "bf16[32768, 256]" = torch.ops.prims.convert_element_type.default(add_tensor, torch.bfloat16);  add_tensor = None
        return (reshape_default, convert_element_type_default_2)


def _default_make_inputs():
    return [
    torch.randn([], dtype=torch.bfloat16, device='cuda'),
    [32768, 256],  # _shape_param_0
    torch.randn([32768, 256], dtype=torch.bfloat16, device='cuda'),
    torch.randn([32768, 1], dtype=torch.float32, device='cuda'),
    torch.randn([256], dtype=torch.float32, device='cuda'),
    [256],  # _shape_param_1
    [32768, 256],  # _shape_param_2
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
