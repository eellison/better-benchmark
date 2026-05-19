"""
Standalone repro captured via capture_hook.
Label: inductor_torchbench_perf-1-6-linux.aws.a100_graph22
Pattern hash: 724c0ac430ab
Shape hash: 44e6f5d5
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
    def forward(self, addmm_71: "bf16[512, 768]", add_81: "bf16[4, 128, 768]", arg197_1: "bf16[2, 768]", arg199_1: "bf16[20005, 768]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        view_default: "bf16[4, 128, 768]" = torch.ops.aten.view.default(addmm_71, _shape_param_0);  addmm_71 = _shape_param_0 = None
        add_tensor: "bf16[4, 128, 768]" = torch.ops.aten.add.Tensor(add_81, view_default);  add_81 = view_default = None
        select_int: "bf16[4, 768]" = torch.ops.aten.select.int(add_tensor, 1, 0)
        permute_default: "bf16[768, 2]" = torch.ops.aten.permute.default(arg197_1, [1, 0]);  arg197_1 = None
        view_default_1: "bf16[512, 768]" = torch.ops.aten.view.default(add_tensor, _shape_param_1);  add_tensor = _shape_param_1 = None
        permute_default_1: "bf16[768, 20005]" = torch.ops.aten.permute.default(arg199_1, [1, 0]);  arg199_1 = None
        return (select_int, permute_default, view_default_1, permute_default_1)


def _default_make_inputs():
    return [
    torch.randn([512, 768], dtype=torch.bfloat16, device='cuda'),
    torch.randn([4, 128, 768], dtype=torch.bfloat16, device='cuda'),
    torch.randn([2, 768], dtype=torch.bfloat16, device='cuda'),
    torch.randn([20005, 768], dtype=torch.bfloat16, device='cuda'),
    [4, 128, 768],  # _shape_param_0
    [512, 768],  # _shape_param_1
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
