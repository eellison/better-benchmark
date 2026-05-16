"""
Standalone repro captured via capture_hook.
Label: tlparse_huggingface_s5_g11
Pattern hash: a852fda9fbc1
Shape hash: 474f2682
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, mm_111: "bf16[1000, 2048]", add_111: "bf16[1, 1000, 2048]", arg147_1: "bf16[2048]", arg1_1: "bf16[128256, 2048]"):
        # No stacktrace found for following nodes
        reshape_default: "bf16[1, 1000, 2048]" = torch.ops.aten.reshape.default(mm_111, [1, 1000, 2048]);  mm_111 = None
        add_tensor: "bf16[1, 1000, 2048]" = torch.ops.aten.add.Tensor(add_111, reshape_default);  add_111 = reshape_default = None
        convert_element_type_default: "f32[1, 1000, 2048]" = torch.ops.prims.convert_element_type.default(add_tensor, torch.float32);  add_tensor = None
        pow_tensor_scalar: "f32[1, 1000, 2048]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_default, 2)
        mean_dim: "f32[1, 1000, 1]" = torch.ops.aten.mean.dim(pow_tensor_scalar, [-1], True);  pow_tensor_scalar = None
        add_tensor_1: "f32[1, 1000, 1]" = torch.ops.aten.add.Tensor(mean_dim, 1e-05);  mean_dim = None
        rsqrt_default: "f32[1, 1000, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        mul_tensor: "f32[1, 1000, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_default, rsqrt_default);  convert_element_type_default = rsqrt_default = None
        convert_element_type_default_1: "bf16[1, 1000, 2048]" = torch.ops.prims.convert_element_type.default(mul_tensor, torch.bfloat16);  mul_tensor = None
        mul_tensor_1: "bf16[1, 1000, 2048]" = torch.ops.aten.mul.Tensor(arg147_1, convert_element_type_default_1);  arg147_1 = convert_element_type_default_1 = None
        permute_default: "bf16[2048, 128256]" = torch.ops.aten.permute.default(arg1_1, [1, 0]);  arg1_1 = None
        reshape_default_1: "bf16[1000, 2048]" = torch.ops.aten.reshape.default(mul_tensor_1, [1000, 2048]);  mul_tensor_1 = None
        return (permute_default, reshape_default_1)


def _default_make_inputs():
    return [
    torch.randn([1000, 2048], dtype=torch.bfloat16, device='cuda'),
    torch.randn([1, 1000, 2048], dtype=torch.bfloat16, device='cuda'),
    torch.randn([2048], dtype=torch.bfloat16, device='cuda'),
    torch.randn([128256, 2048], dtype=torch.bfloat16, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
