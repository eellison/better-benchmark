"""
Standalone repro captured via capture_hook.
Label: torchbench_BERT_pytorch_train
Pattern hash: e262d057f3c9
Shape hash: 2810a40e
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
_shapes_config = "(T([16384, 20005], f32), S([20005]))"

class Repro(torch.nn.Module):
    def forward(self, view_266: "f32[16384, 20005]", _shape_param_0):
        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/language_model.py:61 in forward, code: return self.softmax(self.linear(x))
        sum_dim_int_list: "f32[1, 20005]" = torch.ops.aten.sum.dim_IntList(view_266, [0], True);  view_266 = None
        reshape_default: "f32[20005]" = torch.ops.aten.reshape.default(sum_dim_int_list, _shape_param_0);  sum_dim_int_list = _shape_param_0 = None
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
