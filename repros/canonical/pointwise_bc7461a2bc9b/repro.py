"""
Standalone repro captured via capture_hook.
Label: inductor_timm_perf_cuda_h100-6-7-linux.aws.h100_graph58
Pattern hash: bc7461a2bc9b
Shape hash: ea8e7bb7
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
    def forward(self, getitem_137: "bf16[32, 4, 16, 64]", arg253_1: "bf16[240, 240]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        slice_tensor: "bf16[32, 4, 16, 60]" = torch.ops.aten.slice.Tensor(getitem_137, -1, 0, 60);  getitem_137 = None
        permute_default: "bf16[32, 16, 4, 60]" = torch.ops.aten.permute.default(slice_tensor, [0, 2, 1, 3]);  slice_tensor = None
        clone_default: "bf16[32, 16, 4, 60]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        view_default: "bf16[32, 16, 240]" = torch.ops.aten.view.default(clone_default, _shape_param_0);  clone_default = _shape_param_0 = None
        view_default_1: "bf16[512, 240]" = torch.ops.aten.view.default(view_default, _shape_param_1);  view_default = _shape_param_1 = None
        permute_default_1: "bf16[240, 240]" = torch.ops.aten.permute.default(arg253_1, [1, 0]);  arg253_1 = None
        return (view_default_1, permute_default_1)


def _default_make_inputs():
    return [
    torch.randn([32, 4, 16, 64], dtype=torch.bfloat16, device='cuda'),
    torch.randn([240, 240], dtype=torch.bfloat16, device='cuda'),
    [32, 16, 240],  # _shape_param_0
    [512, 240],  # _shape_param_1
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
