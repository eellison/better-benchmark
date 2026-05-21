"""
Standalone repro captured via capture_hook.
Label: torchbench_nvidia_deeprecommender_infer
Pattern hash: 298ea8f11903
Shape hash: 3e8393fc
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
_shapes_config = "(T([1024, 197951], f32))"

class Repro(torch.nn.Module):
    def forward(self, addmm_5: "f32[1024, 197951]"):
        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nvidia_deeprecommender/reco_encoder/model/model.py:12 in activation, code: return F.selu(input)
        gt_scalar: "b8[1024, 197951]" = torch.ops.aten.gt.Scalar(addmm_5, 0)
        mul_tensor: "f32[1024, 197951]" = torch.ops.aten.mul.Tensor(addmm_5, 1.0507009873554805)
        mul_tensor_1: "f32[1024, 197951]" = torch.ops.aten.mul.Tensor(addmm_5, 1.0);  addmm_5 = None
        expm1_default: "f32[1024, 197951]" = torch.ops.aten.expm1.default(mul_tensor_1);  mul_tensor_1 = None
        mul_tensor_2: "f32[1024, 197951]" = torch.ops.aten.mul.Tensor(expm1_default, 1.7580993408473766);  expm1_default = None
        where_self: "f32[1024, 197951]" = torch.ops.aten.where.self(gt_scalar, mul_tensor, mul_tensor_2);  gt_scalar = mul_tensor = mul_tensor_2 = None
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
