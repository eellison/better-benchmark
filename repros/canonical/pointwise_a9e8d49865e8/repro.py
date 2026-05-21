"""
Standalone repro captured via capture_hook.
Label: torchbench_modded_nanogpt_train
Pattern hash: a9e8d49865e8
Shape hash: 54de7dd5
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
_shapes_config = "(T([50304, 768], torch.float8_e4m3fn))"

class Repro(torch.nn.Module):
    def forward(self, arg1_1: "f8e4m3fn[50304, 768]"):
        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:85 in impl, code: w_f8.T.contiguous().T,
        permute_default: "f8e4m3fn[768, 50304]" = torch.ops.aten.permute.default(arg1_1, [1, 0]);  arg1_1 = None
        clone_default: "f8e4m3fn[768, 50304]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        permute_default_1: "f8e4m3fn[50304, 768]" = torch.ops.aten.permute.default(clone_default, [1, 0]);  clone_default = None
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
