"""
Standalone repro captured via capture_hook.
Label: torchbench_llama_infer
Pattern hash: 57474154bd32
Shape hash: c321495e
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
    def forward(self, mm_52: "f32[1024, 512]", add_34: "f32[32, 32, 512]", arg87_1: "f32[512]", arg88_1: "f32[1536, 512]", arg89_1: "f32[1536, 512]", _shape_param_0, _shape_param_1, _shape_param_2):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/utils/_device.py:122 in __torch_function__, code: return func(*args, **kwargs)
        reshape_default: "f32[32, 32, 512]" = torch.ops.aten.reshape.default(mm_52, _shape_param_0);  mm_52 = _shape_param_0 = None
        add_tensor: "f32[32, 32, 512]" = torch.ops.aten.add.Tensor(add_34, reshape_default);  add_34 = reshape_default = None
        pow_tensor_scalar: "f32[32, 32, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_tensor, 2)
        mean_dim: "f32[32, 32, 1]" = torch.ops.aten.mean.dim(pow_tensor_scalar, [-1], True);  pow_tensor_scalar = None
        add_tensor_1: "f32[32, 32, 1]" = torch.ops.aten.add.Tensor(mean_dim, 1e-05);  mean_dim = None
        rsqrt_default: "f32[32, 32, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        mul_tensor: "f32[32, 32, 512]" = torch.ops.aten.mul.Tensor(add_tensor, rsqrt_default);  add_tensor = rsqrt_default = None
        mul_tensor_1: "f32[32, 32, 512]" = torch.ops.aten.mul.Tensor(mul_tensor, arg87_1);  mul_tensor = arg87_1 = None
        reshape_default_1: "f32[1024, 512]" = torch.ops.aten.reshape.default(mul_tensor_1, _shape_param_1);  _shape_param_1 = None
        permute_default: "f32[512, 1536]" = torch.ops.aten.permute.default(arg88_1, [1, 0]);  arg88_1 = None
        reshape_default_2: "f32[1024, 512]" = torch.ops.aten.reshape.default(mul_tensor_1, _shape_param_2);  mul_tensor_1 = _shape_param_2 = None
        permute_default_1: "f32[512, 1536]" = torch.ops.aten.permute.default(arg89_1, [1, 0]);  arg89_1 = None
        return (reshape_default_1, permute_default, reshape_default_2, permute_default_1)


def _default_make_inputs():
    return []


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
