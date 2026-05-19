"""
Standalone repro captured via capture_hook.
Label: inductor_huggingface_perf-5-5-linux.aws.a100_graph11
Pattern hash: 1d52a0b8ac65
Shape hash: f521085b
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
    def forward(self, bmm_71: "bf16[16, 512, 64]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        view_default: "bf16[16, 512, 1, 1, 64]" = torch.ops.aten.view.default(bmm_71, _shape_param_0);  bmm_71 = _shape_param_0 = None
        permute_default: "bf16[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_default, [1, 3, 0, 4, 2]);  view_default = None
        view_default_1: "bf16[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_default, _shape_param_1);  permute_default = _shape_param_1 = None
        unsqueeze_default: "bf16[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_default_1, 4);  view_default_1 = None
        permute_default_1: "bf16[512, 1, 1, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_default, [0, 1, 4, 3, 2]);  unsqueeze_default = None
        permute_default_2: "bf16[512, 64, 16, 1, 1]" = torch.ops.aten.permute.default(permute_default_1, [0, 3, 4, 1, 2]);  permute_default_1 = None
        clone_default: "bf16[512, 64, 16, 1, 1]" = torch.ops.aten.clone.default(permute_default_2, memory_format = torch.contiguous_format);  permute_default_2 = None
        view_default_2: "bf16[1, 512, 1024]" = torch.ops.aten.view.default(clone_default, _shape_param_2);  clone_default = _shape_param_2 = None
        squeeze_dim: "bf16[512, 1024]" = torch.ops.aten.squeeze.dim(view_default_2, 0);  view_default_2 = None
        return squeeze_dim


def _default_make_inputs():
    return [
    torch.randn([16, 512, 64], dtype=torch.bfloat16, device='cuda'),
    [16, 512, 1, 1, 64],  # _shape_param_0
    [512, 1, 16, 64],  # _shape_param_1
    [1, 512, 1024],  # _shape_param_2
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
