"""
Standalone repro captured via capture_hook.
Label: torchbench_hf_Reformer_train_002
Pattern hash: ff349519eab3
Shape hash: af303225
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([8, 4096, 256], f32), S([8, 4096, 2, 256]), S([8, 4096, 512]))"

class Repro(torch.nn.Module):
    def forward(self, arg0_1: "f32[8, 4096, 256]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        unsqueeze_default: "f32[8, 4096, 1, 256]" = torch.ops.aten.unsqueeze.default(arg0_1, 2);  arg0_1 = None
        expand_default: "f32[8, 4096, 2, 256]" = torch.ops.aten.expand.default(unsqueeze_default, _shape_param_0);  unsqueeze_default = _shape_param_0 = None
        clone_default: "f32[8, 4096, 2, 256]" = torch.ops.aten.clone.default(expand_default, memory_format = torch.contiguous_format);  expand_default = None
        view_default: "f32[8, 4096, 512]" = torch.ops.aten.view.default(clone_default, _shape_param_1);  clone_default = _shape_param_1 = None
        return view_default



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
