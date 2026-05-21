"""
Standalone repro captured via capture_hook.
Label: torchbench_llava_infer
Pattern hash: 45fcdbe93e8a
Shape hash: 47e8d80c
"""
import sys
from pathlib import Path

import sys
from pathlib import Path
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, mm_216: "f16[512, 4096]", add_216: "f16[1, 512, 4096]", arg282_1: "f16[4096]", arg283_1: "f16[4096, 4096]", arg284_1: "f16[4096, 4096]", arg285_1: "f16[4096, 4096]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/utils/_device.py:122 in __torch_function__, code: return func(*args, **kwargs)
        reshape_default: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_216, _shape_param_0);  mm_216 = _shape_param_0 = None
        add_tensor: "f16[1, 512, 4096]" = torch.ops.aten.add.Tensor(add_216, reshape_default);  add_216 = reshape_default = None
        convert_element_type_default: "f32[1, 512, 4096]" = torch.ops.prims.convert_element_type.default(add_tensor, torch.float32);  add_tensor = None
        pow_tensor_scalar: "f32[1, 512, 4096]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_default, 2)
        mean_dim: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(pow_tensor_scalar, [-1], True);  pow_tensor_scalar = None
        add_tensor_1: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(mean_dim, 1e-06);  mean_dim = None
        rsqrt_default: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        mul_tensor: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_default, rsqrt_default);  convert_element_type_default = rsqrt_default = None
        convert_element_type_default_1: "f16[1, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_tensor, torch.float16);  mul_tensor = None
        mul_tensor_1: "f16[1, 512, 4096]" = torch.ops.aten.mul.Tensor(arg282_1, convert_element_type_default_1);  arg282_1 = convert_element_type_default_1 = None
        reshape_default_1: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_tensor_1, _shape_param_1);  _shape_param_1 = None
        permute_default: "f16[4096, 4096]" = torch.ops.aten.permute.default(arg283_1, [1, 0]);  arg283_1 = None
        reshape_default_2: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_tensor_1, _shape_param_2);  _shape_param_2 = None
        permute_default_1: "f16[4096, 4096]" = torch.ops.aten.permute.default(arg284_1, [1, 0]);  arg284_1 = None
        reshape_default_3: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_tensor_1, _shape_param_3);  mul_tensor_1 = _shape_param_3 = None
        permute_default_2: "f16[4096, 4096]" = torch.ops.aten.permute.default(arg285_1, [1, 0]);  arg285_1 = None
        return (reshape_default_1, permute_default, reshape_default_2, permute_default_1, reshape_default_3, permute_default_2)


def _default_make_inputs():
    return []


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
