"""
Standalone repro captured via capture_hook.
Label: torchbench_BERT_pytorch_train
Pattern hash: 1b26ae9b7c91
Shape hash: 66cec495
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
_shapes_config = "(T([128, 2], f32), T([128, 2], f32))"

class Repro(torch.nn.Module):
    def forward(self, sub_37: "f32[128, 2]", tangents_1: "f32[128, 2]"):
        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/language_model.py:42 in forward, code: return self.softmax(self.linear(x[:, 0]))
        exp_default: "f32[128, 2]" = torch.ops.aten.exp.default(sub_37);  sub_37 = None
        sum_dim_int_list: "f32[128, 1]" = torch.ops.aten.sum.dim_IntList(tangents_1, [-1], True)
        mul_tensor: "f32[128, 2]" = torch.ops.aten.mul.Tensor(exp_default, sum_dim_int_list);  exp_default = sum_dim_int_list = None
        sub_tensor: "f32[128, 2]" = torch.ops.aten.sub.Tensor(tangents_1, mul_tensor);  tangents_1 = mul_tensor = None
        return sub_tensor



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
