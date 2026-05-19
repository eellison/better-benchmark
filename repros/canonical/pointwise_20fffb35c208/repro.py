"""
Standalone repro captured via capture_hook.
Label: inductor_huggingface_perf-4-5-linux.aws.a100_graph10
Pattern hash: 20fffb35c208
Shape hash: 5107bfe4
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
    def forward(self, bmm_23: "bf16[16, 256, 64]", arg187_1: "bf16[1024, 1024]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        view_default: "bf16[1, 16, 256, 64]" = torch.ops.aten.view.default(bmm_23, _shape_param_0);  bmm_23 = _shape_param_0 = None
        permute_default: "bf16[1, 256, 16, 64]" = torch.ops.aten.permute.default(view_default, [0, 2, 1, 3]);  view_default = None
        clone_default: "bf16[1, 256, 16, 64]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        view_default_1: "bf16[1, 256, 1024]" = torch.ops.aten.view.default(clone_default, _shape_param_1);  clone_default = _shape_param_1 = None
        view_default_2: "bf16[256, 1024]" = torch.ops.aten.view.default(view_default_1, _shape_param_2);  view_default_1 = _shape_param_2 = None
        permute_default_1: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg187_1, [1, 0]);  arg187_1 = None
        return (view_default_2, permute_default_1)


def _default_make_inputs():
    return [
    torch.randn([16, 256, 64], dtype=torch.bfloat16, device='cuda'),
    torch.randn([1024, 1024], dtype=torch.bfloat16, device='cuda'),
    [1, 16, 256, 64],  # _shape_param_0
    [1, 256, 1024],  # _shape_param_1
    [256, 1024],  # _shape_param_2
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
