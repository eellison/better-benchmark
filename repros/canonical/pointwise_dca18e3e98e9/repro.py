"""
Standalone repro captured via capture_hook.
Label: vllm_Qwen_Qwen3-0.6B_001
Pattern hash: dca18e3e98e9
Shape hash: 7ed52c3f
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
_shapes_config = "(T([4, 16, 512, 128], bf16), S([2048, 2048]))"

class Repro(torch.nn.Module):
    def forward(self, arg863_1: "bf16[4, 16, 512, 128]", _shape_param_0):
        # No stacktrace found for following nodes
        permute_default: "bf16[4, 512, 16, 128]" = torch.ops.aten.permute.default(arg863_1, [0, 2, 1, 3]);  arg863_1 = None
        clone_default: "bf16[4, 512, 16, 128]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        _unsafe_view_default: "bf16[4, 512, 2048]" = torch.ops.aten._unsafe_view.default(clone_default, [4, 512, 2048]);  clone_default = None
        view_default: "bf16[2048, 2048]" = torch.ops.aten.view.default(_unsafe_view_default, _shape_param_0);  _unsafe_view_default = _shape_param_0 = None
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
