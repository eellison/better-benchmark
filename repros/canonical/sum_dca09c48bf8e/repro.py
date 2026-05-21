"""
Standalone repro captured via capture_hook.
Label: torchbench_BERT_pytorch_train
Pattern hash: dca09c48bf8e
Shape hash: 41a01311
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
_shapes_config = "(T([128, 128, 768], f32), S([768]))"

class Repro(torch.nn.Module):
    def forward(self, view_275: "f32[128, 128, 768]", _shape_param_0):
        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17 in forward, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        sum_dim_int_list: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(view_275, [0, 1], True);  view_275 = None
        reshape_default: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list, _shape_param_0);  sum_dim_int_list = _shape_param_0 = None
        return reshape_default



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
