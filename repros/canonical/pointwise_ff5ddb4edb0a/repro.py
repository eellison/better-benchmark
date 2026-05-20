"""
Standalone repro captured via capture_hook.
Label: inductor_huggingface_perf_cuda_h100-4-5-linux.aws.h100_graph39
Pattern hash: ff5ddb4edb0a
Shape hash: 7e4611ca
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([1, 128], i64, max=128), T([2050, 1024], f32), S([1, 128, 1024]))"

class Repro(torch.nn.Module):
    def forward(self, arg0_1: "i64[1, 128]", arg1_1: "f32[2050, 1024]", _shape_param_0):
        # No stacktrace found for following nodes
        add_tensor: "i64[1, 128]" = torch.ops.aten.add.Tensor(arg0_1, 2);  arg0_1 = None
        view_default: "i64[128]" = torch.ops.aten.view.default(add_tensor, [-1]);  add_tensor = None
        index_tensor: "f32[128, 1024]" = torch.ops.aten.index.Tensor(arg1_1, [view_default]);  arg1_1 = view_default = None
        view_default_1: "f32[1, 128, 1024]" = torch.ops.aten.view.default(index_tensor, _shape_param_0);  index_tensor = _shape_param_0 = None
        return view_default_1


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
