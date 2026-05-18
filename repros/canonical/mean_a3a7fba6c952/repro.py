"""
Standalone repro captured via capture_hook.
Label: tlparse_huggingface_s5_g11
Pattern hash: a3a7fba6c952
Shape hash: 9a4a9ecf
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
    def forward(self, mm_104: "bf16[1000, 2048]", add_104: "bf16[1, 1000, 2048]", arg138_1: "bf16[2048]", arg139_1: "bf16[2048, 2048]", arg140_1: "bf16[512, 2048]", arg141_1: "bf16[512, 2048]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # No stacktrace found for following nodes
        reshape_default: "bf16[1, 1000, 2048]" = torch.ops.aten.reshape.default(mm_104, _shape_param_0);  mm_104 = _shape_param_0 = None
        add_tensor: "bf16[1, 1000, 2048]" = torch.ops.aten.add.Tensor(add_104, reshape_default);  add_104 = reshape_default = None
        convert_element_type_default: "f32[1, 1000, 2048]" = torch.ops.prims.convert_element_type.default(add_tensor, torch.float32);  add_tensor = None
        pow_tensor_scalar: "f32[1, 1000, 2048]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_default, 2)
        mean_dim: "f32[1, 1000, 1]" = torch.ops.aten.mean.dim(pow_tensor_scalar, [-1], True);  pow_tensor_scalar = None
        add_tensor_1: "f32[1, 1000, 1]" = torch.ops.aten.add.Tensor(mean_dim, 1e-05);  mean_dim = None
        rsqrt_default: "f32[1, 1000, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        mul_tensor: "f32[1, 1000, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_default, rsqrt_default);  convert_element_type_default = rsqrt_default = None
        convert_element_type_default_1: "bf16[1, 1000, 2048]" = torch.ops.prims.convert_element_type.default(mul_tensor, torch.bfloat16);  mul_tensor = None
        mul_tensor_1: "bf16[1, 1000, 2048]" = torch.ops.aten.mul.Tensor(arg138_1, convert_element_type_default_1);  arg138_1 = convert_element_type_default_1 = None
        permute_default: "bf16[2048, 2048]" = torch.ops.aten.permute.default(arg139_1, [1, 0]);  arg139_1 = None
        reshape_default_1: "bf16[1000, 2048]" = torch.ops.aten.reshape.default(mul_tensor_1, _shape_param_1);  _shape_param_1 = None
        permute_default_1: "bf16[2048, 512]" = torch.ops.aten.permute.default(arg140_1, [1, 0]);  arg140_1 = None
        reshape_default_2: "bf16[1000, 2048]" = torch.ops.aten.reshape.default(mul_tensor_1, _shape_param_2);  _shape_param_2 = None
        permute_default_2: "bf16[2048, 512]" = torch.ops.aten.permute.default(arg141_1, [1, 0]);  arg141_1 = None
        reshape_default_3: "bf16[1000, 2048]" = torch.ops.aten.reshape.default(mul_tensor_1, _shape_param_3);  mul_tensor_1 = _shape_param_3 = None
        return (permute_default, reshape_default_1, permute_default_1, reshape_default_2, permute_default_2, reshape_default_3)


def _default_make_inputs():
    return [
    torch.randn([1000, 2048], dtype=torch.bfloat16, device='cuda'),
    torch.randn([1, 1000, 2048], dtype=torch.bfloat16, device='cuda'),
    torch.randn([2048], dtype=torch.bfloat16, device='cuda'),
    torch.randn([2048, 2048], dtype=torch.bfloat16, device='cuda'),
    torch.randn([512, 2048], dtype=torch.bfloat16, device='cuda'),
    torch.randn([512, 2048], dtype=torch.bfloat16, device='cuda'),
    [1, 1000, 2048],  # _shape_param_0
    [1000, 2048],  # _shape_param_1
    [1000, 2048],  # _shape_param_2
    [1000, 2048],  # _shape_param_3
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
