"""
Standalone repro captured via capture_hook.
Label: inductor_torchbench_perf_cuda_h100-8-9-linux.aws.h100_graph74
Pattern hash: ec83137b9e56
Shape hash: 16631cae
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, bmm_127: "f64[16, 4096, 80]", arg444_1: "f64[1280, 1280]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # No stacktrace found for following nodes
        view_default: "f64[1, 16, 4096, 80]" = torch.ops.aten.view.default(bmm_127, _shape_param_0);  bmm_127 = _shape_param_0 = None
        view_default_1: "f64[1, 16, 64, 64, 80]" = torch.ops.aten.view.default(view_default, _shape_param_1);  view_default = _shape_param_1 = None
        permute_default: "f64[1, 64, 64, 16, 80]" = torch.ops.aten.permute.default(view_default_1, [0, 2, 3, 1, 4]);  view_default_1 = None
        clone_default: "f64[1, 64, 64, 16, 80]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        view_default_2: "f64[1, 64, 64, 1280]" = torch.ops.aten.view.default(clone_default, _shape_param_2);  clone_default = _shape_param_2 = None
        view_default_3: "f64[4096, 1280]" = torch.ops.aten.view.default(view_default_2, _shape_param_3);  view_default_2 = _shape_param_3 = None
        permute_default_1: "f64[1280, 1280]" = torch.ops.aten.permute.default(arg444_1, [1, 0]);  arg444_1 = None
        return (view_default_3, permute_default_1)


def _default_make_inputs():
    return [
    torch.randn([16, 4096, 80], dtype=torch.float64, device='cuda'),
    torch.randn([1280, 1280], dtype=torch.float64, device='cuda'),
    [1, 16, 4096, 80],  # _shape_param_0
    [1, 16, 64, 64, -1],  # _shape_param_1
    [1, 64, 64, 1280],  # _shape_param_2
    [4096, 1280],  # _shape_param_3
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
