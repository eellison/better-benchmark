"""
Standalone repro captured via capture_hook.
Label: inductor_huggingface_perf_cuda_h100-4-5-linux.aws.h100_graph43
Pattern hash: e75e2d1f8e7c
Shape hash: f5b58218
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
    def forward(self, mm_1: "f16[128, 1024]", mm: "f16[256008, 1024]", _shape_param_0):
        # No stacktrace found for following nodes
        view_default: "f16[1, 128, 1024]" = torch.ops.aten.view.default(mm_1, _shape_param_0);  mm_1 = _shape_param_0 = None
        convert_element_type_default: "f32[1, 128, 1024]" = torch.ops.prims.convert_element_type.default(view_default, torch.float32);  view_default = None
        convert_element_type_default_1: "f32[256008, 1024]" = torch.ops.prims.convert_element_type.default(mm, torch.float32);  mm = None
        return (convert_element_type_default, convert_element_type_default_1)


def _default_make_inputs():
    return [
    torch.randn([128, 1024], dtype=torch.float16, device='cuda'),
    torch.randn([256008, 1024], dtype=torch.float16, device='cuda'),
    [1, 128, 1024],  # _shape_param_0
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
