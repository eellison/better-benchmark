"""
Standalone repro captured via capture_hook.
Label: torchbench_modded_nanogpt_train
Pattern hash: 43877c71f0eb
Shape hash: a8fc7cf2
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
_shapes_config = "(T([6144, 768], bf16))"

class Repro(torch.nn.Module):
    def forward(self, arg0_1: "bf16[6144, 768]"):
        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:48 in impl, code: x_f8 = x.div(x_s).to(torch.float8_e4m3fn)
        div_tensor: "bf16[6144, 768]" = torch.ops.aten.div.Tensor(arg0_1, 0.06185895741317419);  arg0_1 = None
        convert_element_type_default: "f8e4m3fn[6144, 768]" = torch.ops.prims.convert_element_type.default(div_tensor, torch.float8_e4m3fn);  div_tensor = None
        return convert_element_type_default



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
