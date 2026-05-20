"""
Standalone repro captured via capture_hook.
Label: inductor_torchbench_perf_cuda_h100-9-9-linux.aws.h100_graph75
Pattern hash: c8175f2f828b
Shape hash: 166aa1b2
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([4, 1000, 13, 13], f16), T([4, 256, 13, 13], f16), T([4, 256, 13, 13], f16), T([4, 256, 13, 13], f16), T([4, 256, 13, 13], f16), T([4, 192, 13, 13], f16), T([4, 192, 13, 13], f16), T([4, 192, 13, 13], f16), T([4, 192, 13, 13], f16), T([4, 128, 27, 27], f16), T([4, 128, 27, 27], f16), T([4, 128, 27, 27], f16), T([4, 128, 27, 27], f16), T([4, 64, 55, 55], f16), T([4, 64, 55, 55], f16), T([4, 64, 55, 55], f16), T([4, 64, 55, 55], f16), S([4, 1000]))"

class Repro(torch.nn.Module):
    def forward(self, convolution_25: "f16[4, 1000, 13, 13]", relu_24: "f16[4, 256, 13, 13]", relu_23: "f16[4, 256, 13, 13]", relu_21: "f16[4, 256, 13, 13]", relu_20: "f16[4, 256, 13, 13]", relu_18: "f16[4, 192, 13, 13]", relu_17: "f16[4, 192, 13, 13]", relu_15: "f16[4, 192, 13, 13]", relu_14: "f16[4, 192, 13, 13]", relu_12: "f16[4, 128, 27, 27]", relu_11: "f16[4, 128, 27, 27]", relu_9: "f16[4, 128, 27, 27]", relu_8: "f16[4, 128, 27, 27]", relu_6: "f16[4, 64, 55, 55]", relu_5: "f16[4, 64, 55, 55]", relu_3: "f16[4, 64, 55, 55]", relu_2: "f16[4, 64, 55, 55]", _shape_param_0):
        # No stacktrace found for following nodes
        relu_default: "f16[4, 1000, 13, 13]" = torch.ops.aten.relu.default(convolution_25);  convolution_25 = None
        mean_dim: "f16[4, 1000, 1, 1]" = torch.ops.aten.mean.dim(relu_default, [-1, -2], True)
        view_default: "f16[4, 1000]" = torch.ops.aten.view.default(mean_dim, _shape_param_0);  mean_dim = _shape_param_0 = None
        le_scalar: "b8[4, 1000, 13, 13]" = torch.ops.aten.le.Scalar(relu_default, 0);  relu_default = None
        le_scalar_1: "b8[4, 256, 13, 13]" = torch.ops.aten.le.Scalar(relu_24, 0);  relu_24 = None
        le_scalar_2: "b8[4, 256, 13, 13]" = torch.ops.aten.le.Scalar(relu_23, 0);  relu_23 = None
        le_scalar_3: "b8[4, 256, 13, 13]" = torch.ops.aten.le.Scalar(relu_21, 0);  relu_21 = None
        le_scalar_4: "b8[4, 256, 13, 13]" = torch.ops.aten.le.Scalar(relu_20, 0);  relu_20 = None
        le_scalar_5: "b8[4, 192, 13, 13]" = torch.ops.aten.le.Scalar(relu_18, 0);  relu_18 = None
        le_scalar_6: "b8[4, 192, 13, 13]" = torch.ops.aten.le.Scalar(relu_17, 0);  relu_17 = None
        le_scalar_7: "b8[4, 192, 13, 13]" = torch.ops.aten.le.Scalar(relu_15, 0);  relu_15 = None
        le_scalar_8: "b8[4, 192, 13, 13]" = torch.ops.aten.le.Scalar(relu_14, 0);  relu_14 = None
        le_scalar_9: "b8[4, 128, 27, 27]" = torch.ops.aten.le.Scalar(relu_12, 0);  relu_12 = None
        le_scalar_10: "b8[4, 128, 27, 27]" = torch.ops.aten.le.Scalar(relu_11, 0);  relu_11 = None
        le_scalar_11: "b8[4, 128, 27, 27]" = torch.ops.aten.le.Scalar(relu_9, 0);  relu_9 = None
        le_scalar_12: "b8[4, 128, 27, 27]" = torch.ops.aten.le.Scalar(relu_8, 0);  relu_8 = None
        le_scalar_13: "b8[4, 64, 55, 55]" = torch.ops.aten.le.Scalar(relu_6, 0);  relu_6 = None
        le_scalar_14: "b8[4, 64, 55, 55]" = torch.ops.aten.le.Scalar(relu_5, 0);  relu_5 = None
        le_scalar_15: "b8[4, 64, 55, 55]" = torch.ops.aten.le.Scalar(relu_3, 0);  relu_3 = None
        le_scalar_16: "b8[4, 64, 55, 55]" = torch.ops.aten.le.Scalar(relu_2, 0);  relu_2 = None
        return (view_default, le_scalar, le_scalar_1, le_scalar_2, le_scalar_3, le_scalar_4, le_scalar_5, le_scalar_6, le_scalar_7, le_scalar_8, le_scalar_9, le_scalar_10, le_scalar_11, le_scalar_12, le_scalar_13, le_scalar_14, le_scalar_15, le_scalar_16)


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
