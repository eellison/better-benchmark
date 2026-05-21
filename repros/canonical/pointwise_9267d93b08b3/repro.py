"""
Standalone repro captured via capture_hook.
Label: hf_GPTJForCausalLM_train_001
Pattern hash: 9267d93b08b3
Shape hash: 9037b713
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
_shapes_config = "(T([1, 16, 128, 256], f32, stride=(524288, 256, 4096, 1)), S([1, 128, 4096]), S([128, 4096]))"

class Repro(torch.nn.Module):
    def forward(self, getitem_110: "f32[1, 16, 128, 256]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        permute_default: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(getitem_110, [0, 2, 1, 3]);  getitem_110 = None
        clone_default: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        view_default: "f32[1, 128, 4096]" = torch.ops.aten.view.default(clone_default, _shape_param_0);  clone_default = _shape_param_0 = None
        view_default_1: "f32[128, 4096]" = torch.ops.aten.view.default(view_default, _shape_param_1);  view_default = _shape_param_1 = None
        permute_default_1: "f32[4096, 128]" = torch.ops.aten.permute.default(view_default_1, [1, 0]);  view_default_1 = None
        return permute_default_1



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
