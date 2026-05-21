"""
Standalone repro captured via capture_hook.
Label: torchbench_dlrm_train
Pattern hash: e00c80fe242e
Shape hash: 3997c37d
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
_shapes_config = "(T([2048, 1], f32))"

class Repro(torch.nn.Module):
    def forward(self, where: "f32[2048, 1]"):
        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/dlrm/dlrm_s_pytorch.py:284 in apply_mlp, code: return layers(x)
        sum_dim_int_list: "f32[1, 1]" = torch.ops.aten.sum.dim_IntList(where, [0], True);  where = None
        reshape_default: "f32[1]" = torch.ops.aten.reshape.default(sum_dim_int_list, [1]);  sum_dim_int_list = None
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
