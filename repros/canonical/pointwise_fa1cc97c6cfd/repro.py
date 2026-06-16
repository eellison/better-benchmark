"""
Standalone repro captured via capture_hook.
Label: hf_BartForCausalLM_train
Pattern hash: fa1cc97c6cfd
Shape hash: 0473776d
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 3
# Input shapes/strides/dtypes live in the sibling shapes.json (structured,
# one entry per point); forward()'s annotations document the default shapes
# inline. Default inputs = the first shapes.json point.

class Repro(torch.nn.Module):
    def forward(self, arg0_1: "bf16[50272, 1024]"):
        # No stacktrace found for following nodes
        slice_1: "bf16[50265, 1024]" = torch.ops.aten.slice.Tensor(arg0_1, 0, 0, -7);  arg0_1 = None
        convert_element_type: "f32[50265, 1024]" = torch.ops.prims.convert_element_type.default(slice_1, torch.float32);  slice_1 = None
        return convert_element_type



def _default_make_inputs():
    configs = load_shape_configs(__file__)
    if not configs:
        raise RuntimeError(
            "no shapes.json next to this repro — pass an explicit config "
            "via make_inputs(shape_config=...)")
    return make_inputs_from_config(next(iter(configs.values())))


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
