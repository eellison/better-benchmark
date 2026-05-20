"""
Standalone repro captured via capture_hook.
Label: inductor_torchbench_perf_cuda_h100-8-9-linux.aws.h100_graph74
Pattern hash: f61f1568f00f
Shape hash: 0a9e1ea9
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([4096, 1280], f64), T([1, 64, 64, 1280], f64, stride=(5242880, 64, 1, 4096)), S([1, 64, 64, 1280]))"

class Repro(torch.nn.Module):
    def forward(self, addmm_127: "f64[4096, 1280]", add_348: "f64[1, 64, 64, 1280]", _shape_param_0):
        # No stacktrace found for following nodes
        view_default: "f64[1, 64, 64, 1280]" = torch.ops.aten.view.default(addmm_127, _shape_param_0);  addmm_127 = _shape_param_0 = None
        add_tensor: "f64[1, 64, 64, 1280]" = torch.ops.aten.add.Tensor(add_348, view_default);  add_348 = view_default = None
        permute_default: "f64[1, 1280, 64, 64]" = torch.ops.aten.permute.default(add_tensor, [0, 3, 1, 2]);  add_tensor = None
        return permute_default


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
