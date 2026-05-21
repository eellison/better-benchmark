"""
Standalone repro captured via capture_hook.
Label: torchbench_dlrm_train
Pattern hash: 4abd33d2c9cd
Shape hash: f3cf922e
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
_shapes_config = "(T([2048, 512], f32), T([], f32), T([2048, 512], f32), S([512]))"

class Repro(torch.nn.Module):
    def forward(self, relu: "f32[2048, 512]", full_default: "f32[]", mm_7: "f32[2048, 512]", _shape_param_0):
        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/dlrm/dlrm_s_pytorch.py:284 in apply_mlp, code: return layers(x)
        le_scalar: "b8[2048, 512]" = torch.ops.aten.le.Scalar(relu, 0);  relu = None
        where_self: "f32[2048, 512]" = torch.ops.aten.where.self(le_scalar, full_default, mm_7);  le_scalar = full_default = mm_7 = None
        permute_default: "f32[512, 2048]" = torch.ops.aten.permute.default(where_self, [1, 0])
        sum_dim_int_list: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(where_self, [0], True);  where_self = None
        reshape_default: "f32[512]" = torch.ops.aten.reshape.default(sum_dim_int_list, _shape_param_0);  sum_dim_int_list = _shape_param_0 = None
        return (permute_default, reshape_default)



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
