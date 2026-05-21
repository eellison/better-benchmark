"""
Standalone repro captured via capture_hook.
Label: torchbench_soft_actor_critic_train_000
Pattern hash: 4e4da02d582f
Shape hash: 9f43f75e
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
_shapes_config = "(T([256, 2], f32))"

class Repro(torch.nn.Module):
    def forward(self, addmm_2: "f32[256, 2]"):
        # No stacktrace found for following nodes
        split_tensor = torch.ops.aten.split.Tensor(addmm_2, 1, 1);  addmm_2 = None
        getitem: "f32[256, 1]" = split_tensor[0]
        getitem_1: "f32[256, 1]" = split_tensor[1];  split_tensor = None
        tanh_default: "f32[256, 1]" = torch.ops.aten.tanh.default(getitem_1);  getitem_1 = None
        add_tensor: "f32[256, 1]" = torch.ops.aten.add.Tensor(tanh_default, 1)
        mul_tensor: "f32[256, 1]" = torch.ops.aten.mul.Tensor(add_tensor, 6.0);  add_tensor = None
        add_tensor_1: "f32[256, 1]" = torch.ops.aten.add.Tensor(mul_tensor, -10.0);  mul_tensor = None
        exp_default: "f32[256, 1]" = torch.ops.aten.exp.default(add_tensor_1);  add_tensor_1 = None
        mul_tensor_1: "f32[256, 1]" = torch.ops.aten.mul.Tensor(tanh_default, tanh_default);  tanh_default = None
        sub_tensor: "f32[256, 1]" = torch.ops.aten.sub.Tensor(1, mul_tensor_1);  mul_tensor_1 = None
        return (getitem, exp_default, sub_tensor)



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
