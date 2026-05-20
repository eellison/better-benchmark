"""
Standalone repro captured via capture_hook.
Label: inductor_huggingface_perf-1-5-linux.aws.a100_graph2
Pattern hash: 7b1fc1d67733
Shape hash: 74873c57
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([48, 256, 64], bf16), T([768, 768], bf16), S([12, 4, 256, 1, 64]), S([12, 4, 256, 64]), S([1, 12, 1024, 64]), S([1024, 1, 768]), S([1024, 768]))"

class Repro(torch.nn.Module):
    def forward(self, bmm_23: "bf16[48, 256, 64]", arg185_1: "bf16[768, 768]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4):
        # No stacktrace found for following nodes
        view_default: "bf16[12, 4, 256, 1, 64]" = torch.ops.aten.view.default(bmm_23, _shape_param_0);  bmm_23 = _shape_param_0 = None
        permute_default: "bf16[12, 4, 256, 64, 1]" = torch.ops.aten.permute.default(view_default, [0, 1, 2, 4, 3]);  view_default = None
        view_default_1: "bf16[12, 4, 256, 64]" = torch.ops.aten.view.default(permute_default, _shape_param_1);  permute_default = _shape_param_1 = None
        view_default_2: "bf16[1, 12, 1024, 64]" = torch.ops.aten.view.default(view_default_1, _shape_param_2);  view_default_1 = _shape_param_2 = None
        permute_default_1: "bf16[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_default_2, [0, 2, 1, 3]);  view_default_2 = None
        permute_default_2: "bf16[1024, 1, 12, 64]" = torch.ops.aten.permute.default(permute_default_1, [1, 0, 2, 3]);  permute_default_1 = None
        clone_default: "bf16[1024, 1, 12, 64]" = torch.ops.aten.clone.default(permute_default_2, memory_format = torch.contiguous_format);  permute_default_2 = None
        view_default_3: "bf16[1024, 1, 768]" = torch.ops.aten.view.default(clone_default, _shape_param_3);  clone_default = _shape_param_3 = None
        permute_default_3: "bf16[1, 1024, 768]" = torch.ops.aten.permute.default(view_default_3, [1, 0, 2]);  view_default_3 = None
        view_default_4: "bf16[1024, 768]" = torch.ops.aten.view.default(permute_default_3, _shape_param_4);  permute_default_3 = _shape_param_4 = None
        permute_default_4: "bf16[768, 768]" = torch.ops.aten.permute.default(arg185_1, [1, 0]);  arg185_1 = None
        return (view_default_4, permute_default_4)


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
