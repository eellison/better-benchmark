"""
Standalone repro captured via capture_hook.
Label: inductor_torchbench_perf-1-6-linux.aws.a100_graph22
Pattern hash: bbbfbeba4f58
Shape hash: c9fdfa30
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
    def forward(self, addmm_66: "bf16[512, 768]", addmm_67: "bf16[512, 768]", add_80: "bf16[4, 128, 768]", arg187_1: "bf16[768, 768]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8):
        # No stacktrace found for following nodes
        view_default: "bf16[4, 128, 768]" = torch.ops.aten.view.default(addmm_66, _shape_param_0);  addmm_66 = _shape_param_0 = None
        view_default_1: "bf16[4, 128, 12, 64]" = torch.ops.aten.view.default(view_default, _shape_param_1);  view_default = _shape_param_1 = None
        permute_default: "bf16[4, 12, 128, 64]" = torch.ops.aten.permute.default(view_default_1, [0, 2, 1, 3]);  view_default_1 = None
        view_default_2: "bf16[4, 128, 768]" = torch.ops.aten.view.default(addmm_67, _shape_param_2);  addmm_67 = _shape_param_2 = None
        view_default_3: "bf16[4, 128, 12, 64]" = torch.ops.aten.view.default(view_default_2, _shape_param_3);  view_default_2 = _shape_param_3 = None
        permute_default_1: "bf16[4, 12, 128, 64]" = torch.ops.aten.permute.default(view_default_3, [0, 2, 1, 3]);  view_default_3 = None
        view_default_4: "bf16[512, 768]" = torch.ops.aten.view.default(add_80, _shape_param_4);  add_80 = _shape_param_4 = None
        permute_default_2: "bf16[768, 768]" = torch.ops.aten.permute.default(arg187_1, [1, 0]);  arg187_1 = None
        permute_default_3: "bf16[4, 12, 64, 128]" = torch.ops.aten.permute.default(permute_default_1, [0, 1, 3, 2]);  permute_default_1 = None
        expand_default: "bf16[4, 12, 128, 64]" = torch.ops.aten.expand.default(permute_default, _shape_param_5);  permute_default = _shape_param_5 = None
        clone_default: "bf16[4, 12, 128, 64]" = torch.ops.aten.clone.default(expand_default, memory_format = torch.contiguous_format);  expand_default = None
        view_default_5: "bf16[48, 128, 64]" = torch.ops.aten.view.default(clone_default, _shape_param_6);  clone_default = _shape_param_6 = None
        expand_default_1: "bf16[4, 12, 64, 128]" = torch.ops.aten.expand.default(permute_default_3, _shape_param_7);  permute_default_3 = _shape_param_7 = None
        clone_default_1: "bf16[4, 12, 64, 128]" = torch.ops.aten.clone.default(expand_default_1, memory_format = torch.contiguous_format);  expand_default_1 = None
        view_default_6: "bf16[48, 64, 128]" = torch.ops.aten.view.default(clone_default_1, _shape_param_8);  clone_default_1 = _shape_param_8 = None
        return (view_default_4, permute_default_2, view_default_5, view_default_6)


def _default_make_inputs():
    return [
    torch.randn([512, 768], dtype=torch.bfloat16, device='cuda'),
    torch.randn([512, 768], dtype=torch.bfloat16, device='cuda'),
    torch.randn([4, 128, 768], dtype=torch.bfloat16, device='cuda'),
    torch.randn([768, 768], dtype=torch.bfloat16, device='cuda'),
    [4, 128, 768],  # _shape_param_0
    [4, -1, 12, 64],  # _shape_param_1
    [4, 128, 768],  # _shape_param_2
    [4, -1, 12, 64],  # _shape_param_3
    [512, 768],  # _shape_param_4
    [4, 12, 128, 64],  # _shape_param_5
    [48, 128, 64],  # _shape_param_6
    [4, 12, 64, 128],  # _shape_param_7
    [48, 64, 128],  # _shape_param_8
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
