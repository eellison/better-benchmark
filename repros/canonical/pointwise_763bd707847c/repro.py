"""
Standalone repro captured via capture_hook.
Label: torchbench_modded_nanogpt_infer
Pattern hash: 763bd707847c
Shape hash: 1d228b78
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
_shapes_config = "(T([768, 3072], f32))"

class Repro(torch.nn.Module):
    def forward(self, arg73_1: "f32[768, 3072]"):
        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:723 in forward, code: x = F.linear(x, self.c_fc.T.type_as(x))
        permute_default: "f32[3072, 768]" = torch.ops.aten.permute.default(arg73_1, [1, 0]);  arg73_1 = None
        convert_element_type_default: "bf16[3072, 768]" = torch.ops.prims.convert_element_type.default(permute_default, torch.bfloat16);  permute_default = None
        permute_default_1: "bf16[768, 3072]" = torch.ops.aten.permute.default(convert_element_type_default, [1, 0]);  convert_element_type_default = None
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
