"""
Standalone repro captured via capture_hook.
Label: inductor_huggingface_perf_cuda_h100-4-5-linux.aws.h100_graph42
Pattern hash: 2efa5da28e80
Shape hash: e58f0079
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
    def forward(self, arg1_1: "f32[256008, 1024]", arg0_1: "f32[1, 128, 1024]", _shape_param_0):
        # No stacktrace found for following nodes
        convert_element_type_default: "f16[256008, 1024]" = torch.ops.prims.convert_element_type.default(arg1_1, torch.float16);  arg1_1 = None
        convert_element_type_default_1: "f16[1, 128, 1024]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.float16);  arg0_1 = None
        permute_default: "f16[1024, 256008]" = torch.ops.aten.permute.default(convert_element_type_default, [1, 0]);  convert_element_type_default = None
        view_default: "f16[128, 1024]" = torch.ops.aten.view.default(convert_element_type_default_1, _shape_param_0);  convert_element_type_default_1 = _shape_param_0 = None
        return (permute_default, view_default)


def _default_make_inputs():
    return [
    torch.randn([256008, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 1024], dtype=torch.float32, device='cuda'),
    [128, 1024],  # _shape_param_0
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
