"""
Standalone repro captured via capture_hook.
Label: torchbench_modded_nanogpt_infer
Pattern hash: 54ea289b90ad
Shape hash: adb33088
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
_shapes_config = "(T([6, 12], f32), S([1, 12, 6]))"

class Repro(torch.nn.Module):
    def forward(self, arg72_1: "f32[6, 12]", _shape_param_0):
        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:618 in forward, code: return F.linear(x, self.weight.type_as(x))
        convert_element_type_default: "bf16[6, 12]" = torch.ops.prims.convert_element_type.default(arg72_1, torch.bfloat16);  arg72_1 = None
        permute_default: "bf16[12, 6]" = torch.ops.aten.permute.default(convert_element_type_default, [1, 0]);  convert_element_type_default = None
        expand_default: "bf16[1, 12, 6]" = torch.ops.aten.expand.default(permute_default, _shape_param_0);  permute_default = _shape_param_0 = None
        squeeze_dim: "bf16[12, 6]" = torch.ops.aten.squeeze.dim(expand_default, 0);  expand_default = None
        return squeeze_dim



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
