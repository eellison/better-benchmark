"""
Standalone repro captured via capture_hook.
Label: torchbench_modded_nanogpt_train
Pattern hash: a718233d708d
Shape hash: 90c86a06
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
_shapes_config = "(T([50304, 768], f32))"

class Repro(torch.nn.Module):
    def forward(self, arg1_1: "f32[50304, 768]"):
        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:49 in impl, code: w_f8 = w.div(w_s).to(torch.float8_e4m3fn)
        div_tensor: "f32[50304, 768]" = torch.ops.aten.div.Tensor(arg1_1, 0.001953125);  arg1_1 = None
        convert_element_type_default: "f8e4m3fn[50304, 768]" = torch.ops.prims.convert_element_type.default(div_tensor, torch.float8_e4m3fn);  div_tensor = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:52 in impl, code: w_f8.T,
        permute_default: "f8e4m3fn[768, 50304]" = torch.ops.aten.permute.default(convert_element_type_default, [1, 0]);  convert_element_type_default = None
        return permute_default



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
