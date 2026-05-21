"""
Standalone repro captured via capture_hook.
Label: torchbench_demucs_train_003
Pattern hash: e82795660886
Shape hash: 86275f76
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
_shapes_config = "(T([64, 4, 2, 371372], f32), S([64, 8, 371372]))"

class Repro(torch.nn.Module):
    def forward(self, arg35_1: "f32[64, 4, 2, 371372]", _shape_param_0):
        # No stacktrace found for following nodes
        view_default: "f32[64, 8, 371372]" = torch.ops.aten.view.default(arg35_1, _shape_param_0);  arg35_1 = _shape_param_0 = None
        sum_dim_int_list: "f32[8]" = torch.ops.aten.sum.dim_IntList(view_default, [0, 2]);  view_default = None
        return sum_dim_int_list



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
