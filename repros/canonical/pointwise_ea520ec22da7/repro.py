"""
Standalone repro captured via capture_hook.
Label: inductor_timm_perf-6-6-linux.aws.a100_graph21
Pattern hash: ea520ec22da7
Shape hash: 9ab77262
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
    def forward(self, addmm_44: "bf16[175360, 2304]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        view_default: "bf16[128, 1370, 2304]" = torch.ops.aten.view.default(addmm_44, _shape_param_0);  addmm_44 = _shape_param_0 = None
        view_default_1: "bf16[128, 1370, 3, 12, 64]" = torch.ops.aten.view.default(view_default, _shape_param_1);  view_default = _shape_param_1 = None
        permute_default: "bf16[3, 128, 12, 1370, 64]" = torch.ops.aten.permute.default(view_default_1, [2, 0, 3, 1, 4]);  view_default_1 = None
        unbind_int = torch.ops.aten.unbind.int(permute_default);  permute_default = None
        getitem: "bf16[128, 12, 1370, 64]" = unbind_int[0]
        getitem_1: "bf16[128, 12, 1370, 64]" = unbind_int[1]
        getitem_2: "bf16[128, 12, 1370, 64]" = unbind_int[2];  unbind_int = None
        return (getitem, getitem_1, getitem_2)


def _default_make_inputs():
    return [
    torch.randn([175360, 2304], dtype=torch.bfloat16, device='cuda'),
    [128, 1370, 2304],  # _shape_param_0
    [128, 1370, 3, 12, 64],  # _shape_param_1
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
