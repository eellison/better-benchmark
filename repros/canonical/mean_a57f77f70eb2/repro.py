"""
Standalone repro captured via capture_hook.
Label: torchbench_llama_infer
Pattern hash: a57f77f70eb2
Shape hash: 1596f0e9
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
_shapes_config = "(T([1024, 512], f32), T([32, 32, 512], f32), T([512], f32), S([32, 32, 512]))"

class Repro(torch.nn.Module):
    def forward(self, mm_55: "f32[1024, 512]", add_36: "f32[32, 32, 512]", arg91_1: "f32[512]", _shape_param_0):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/utils/_device.py:122 in __torch_function__, code: return func(*args, **kwargs)
        reshape_default: "f32[32, 32, 512]" = torch.ops.aten.reshape.default(mm_55, _shape_param_0);  mm_55 = _shape_param_0 = None
        add_tensor: "f32[32, 32, 512]" = torch.ops.aten.add.Tensor(add_36, reshape_default);  add_36 = reshape_default = None
        pow_tensor_scalar: "f32[32, 32, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_tensor, 2)
        mean_dim: "f32[32, 32, 1]" = torch.ops.aten.mean.dim(pow_tensor_scalar, [-1], True);  pow_tensor_scalar = None
        add_tensor_1: "f32[32, 32, 1]" = torch.ops.aten.add.Tensor(mean_dim, 1e-05);  mean_dim = None
        rsqrt_default: "f32[32, 32, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        mul_tensor: "f32[32, 32, 512]" = torch.ops.aten.mul.Tensor(add_tensor, rsqrt_default);  add_tensor = rsqrt_default = None
        mul_tensor_1: "f32[32, 32, 512]" = torch.ops.aten.mul.Tensor(mul_tensor, arg91_1);  mul_tensor = arg91_1 = None
        select_int: "f32[32, 512]" = torch.ops.aten.select.int(mul_tensor_1, 1, -1);  mul_tensor_1 = None
        return select_int



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
