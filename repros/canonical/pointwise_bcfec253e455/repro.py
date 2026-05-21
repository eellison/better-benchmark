"""
Standalone repro captured via capture_hook.
Label: torchbench_moco_infer
Pattern hash: bcfec253e455
Shape hash: d5f011c5
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
_shapes_config = "(T([64, 3, 7, 7], f32), T([64, 3, 7, 7], f32))"

class Repro(torch.nn.Module):
    def forward(self, arg1_1: "f32[64, 3, 7, 7]", arg2_1: "f32[64, 3, 7, 7]"):
        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/moco/moco/builder.py:59 in _momentum_update_key_encoder, code: param_k.mul_(self.m).add_(param_q.mul(1.0 - self.m))
        mul_tensor: "f32[64, 3, 7, 7]" = torch.ops.aten.mul.Tensor(arg1_1, 0.999)
        mul_tensor_1: "f32[64, 3, 7, 7]" = torch.ops.aten.mul.Tensor(arg2_1, 0.0010000000000000009);  arg2_1 = None
        add_tensor: "f32[64, 3, 7, 7]" = torch.ops.aten.add.Tensor(mul_tensor, mul_tensor_1);  mul_tensor = mul_tensor_1 = None
        copy__default: "f32[64, 3, 7, 7]" = torch.ops.aten.copy_.default(arg1_1, add_tensor);  arg1_1 = add_tensor = None
        return copy__default



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
