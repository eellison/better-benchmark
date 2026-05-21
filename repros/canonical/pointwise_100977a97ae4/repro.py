"""
Standalone repro captured via capture_hook.
Label: torchbench_nvidia_deeprecommender_train
Pattern hash: 100977a97ae4
Shape hash: e9946717
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
_shapes_config = "(T([1024, 512], f32), T([1024, 512], f32))"

class Repro(torch.nn.Module):
    def forward(self, addmm_1: "f32[1024, 512]", mm_6: "f32[1024, 512]"):
        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nvidia_deeprecommender/reco_encoder/model/model.py:12 in activation, code: return F.selu(input)
        le_scalar: "b8[1024, 512]" = torch.ops.aten.le.Scalar(addmm_1, 0)
        mul_tensor: "f32[1024, 512]" = torch.ops.aten.mul.Tensor(mm_6, 1)
        mul_tensor_1: "f32[1024, 512]" = torch.ops.aten.mul.Tensor(mul_tensor, 1.7580993408473766);  mul_tensor = None
        mul_tensor_2: "f32[1024, 512]" = torch.ops.aten.mul.Tensor(addmm_1, 1);  addmm_1 = None
        exp_default: "f32[1024, 512]" = torch.ops.aten.exp.default(mul_tensor_2);  mul_tensor_2 = None
        mul_tensor_3: "f32[1024, 512]" = torch.ops.aten.mul.Tensor(mul_tensor_1, exp_default);  mul_tensor_1 = exp_default = None
        mul_tensor_4: "f32[1024, 512]" = torch.ops.aten.mul.Tensor(mm_6, 1.0507009873554805);  mm_6 = None
        where_self: "f32[1024, 512]" = torch.ops.aten.where.self(le_scalar, mul_tensor_3, mul_tensor_4);  le_scalar = mul_tensor_3 = mul_tensor_4 = None
        return where_self



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
