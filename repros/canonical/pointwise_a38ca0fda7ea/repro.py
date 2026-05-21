"""
Standalone repro captured via capture_hook.
Label: torchbench_soft_actor_critic_train
Pattern hash: a38ca0fda7ea
Shape hash: d636d51d
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
_shapes_config = "(T([256, 1], f32), T([256, 1], f32), T([256, 1], f32, stride=(2, 1)), T([256, 1], f32, stride=(2, 1)), T([256, 1], f32), T([256, 1], f32))"

class Repro(torch.nn.Module):
    def forward(self, tangents_2: "f32[256, 1]", tangents_3: "f32[256, 1]", tangents_1: "f32[256, 1]", tangents_4: "f32[256, 1]", exp: "f32[256, 1]", sub: "f32[256, 1]"):
        # No stacktrace found for following nodes
        add_tensor: "f32[256, 1]" = torch.ops.aten.add.Tensor(tangents_2, tangents_3);  tangents_2 = tangents_3 = None
        add_tensor_1: "f32[256, 1]" = torch.ops.aten.add.Tensor(tangents_1, tangents_4);  tangents_1 = tangents_4 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/soft_actor_critic/nets.py:122 in forward, code: std = log_std.exp()
        mul_tensor: "f32[256, 1]" = torch.ops.aten.mul.Tensor(add_tensor, exp);  add_tensor = exp = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/soft_actor_critic/nets.py:119 in forward, code: log_std = self.log_std_low + 0.5 * (
        mul_tensor_1: "f32[256, 1]" = torch.ops.aten.mul.Tensor(mul_tensor, 6.0);  mul_tensor = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/soft_actor_critic/nets.py:118 in forward, code: log_std = torch.tanh(log_std)
        mul_tensor_2: "f32[256, 1]" = torch.ops.aten.mul.Tensor(mul_tensor_1, sub);  mul_tensor_1 = sub = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/soft_actor_critic/nets.py:116 in forward, code: mu, log_std = out.chunk(2, dim=1)
        cat_default: "f32[256, 2]" = torch.ops.aten.cat.default([add_tensor_1, mul_tensor_2], 1);  add_tensor_1 = mul_tensor_2 = None
        return cat_default



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
