"""
Standalone repro captured via capture_hook.
Label: torchbench_demucs_train_002
Pattern hash: 0c71173959e2
Shape hash: b3d7a3ec
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
_shapes_config = "(T([64, 64, 92844], f32), T([64, 64, 95696], f32))"

class Repro(torch.nn.Module):
    def forward(self, convolution_9: "f32[64, 64, 92844]", arg26_1: "f32[64, 64, 95696]"):
        # No stacktrace found for following nodes
        relu_default: "f32[64, 64, 92844]" = torch.ops.aten.relu.default(convolution_9);  convolution_9 = None
        slice_tensor: "f32[64, 64, 92844]" = torch.ops.aten.slice.Tensor(arg26_1, 2, 1426, -1426);  arg26_1 = None
        add_tensor: "f32[64, 64, 92844]" = torch.ops.aten.add.Tensor(relu_default, slice_tensor);  slice_tensor = None
        le_scalar: "b8[64, 64, 92844]" = torch.ops.aten.le.Scalar(relu_default, 0);  relu_default = None
        return (add_tensor, le_scalar)



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
