"""
Standalone repro captured via capture_hook.
Label: torchbench_llama_infer_000
Pattern hash: 2cb3b25427b0
Shape hash: 94759a4e
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
_shapes_config = "(T([32, 32], torch.complex64), T([32, 32, 8, 32], torch.complex64), T([32, 32, 8, 32], torch.complex64), S([1, 32, 1, 32]))"

class Repro(torch.nn.Module):
    def forward(self, slice_1: "c64[32, 32]", view_as_complex_14: "c64[32, 32, 8, 32]", view_as_complex_15: "c64[32, 32, 8, 32]", _shape_param_0):
        # No stacktrace found for following nodes
        view_default: "c64[1, 32, 1, 32]" = torch.ops.aten.view.default(slice_1, _shape_param_0);  slice_1 = _shape_param_0 = None
        mul_tensor: "c64[32, 32, 8, 32]" = torch.ops.aten.mul.Tensor(view_as_complex_14, view_default);  view_as_complex_14 = None
        mul_tensor_1: "c64[32, 32, 8, 32]" = torch.ops.aten.mul.Tensor(view_as_complex_15, view_default);  view_as_complex_15 = view_default = None
        return (mul_tensor, mul_tensor_1)



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
