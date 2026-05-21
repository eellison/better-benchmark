"""
Standalone repro captured via capture_hook.
Label: vllm_Qwen_Qwen3-0.6B_000
Pattern hash: 49473bb3f273
Shape hash: 6bfb868a
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
_shapes_config = "(T([2048, 1024], bf16), S([4, 512, 1024]), S([4, 512, -1, 128]), S([4, 8, 2, 512, 128]), S([4, 16, 512, 128]))"

class Repro(torch.nn.Module):
    def forward(self, mm_191: "bf16[2048, 1024]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # No stacktrace found for following nodes
        view_default: "bf16[4, 512, 1024]" = torch.ops.aten.view.default(mm_191, _shape_param_0);  mm_191 = _shape_param_0 = None
        view_default_1: "bf16[4, 512, 8, 128]" = torch.ops.aten.view.default(view_default, _shape_param_1);  view_default = _shape_param_1 = None
        permute_default: "bf16[4, 8, 512, 128]" = torch.ops.aten.permute.default(view_default_1, [0, 2, 1, 3]);  view_default_1 = None
        unsqueeze_default: "bf16[4, 8, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(permute_default, 2);  permute_default = None
        expand_default: "bf16[4, 8, 2, 512, 128]" = torch.ops.aten.expand.default(unsqueeze_default, _shape_param_2);  unsqueeze_default = _shape_param_2 = None
        clone_default: "bf16[4, 8, 2, 512, 128]" = torch.ops.aten.clone.default(expand_default, memory_format = torch.contiguous_format);  expand_default = None
        view_default_2: "bf16[4, 16, 512, 128]" = torch.ops.aten.view.default(clone_default, _shape_param_3);  clone_default = _shape_param_3 = None
        return view_default_2



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
