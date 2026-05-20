"""
Standalone repro captured via capture_hook.
Label: inductor_torchbench_perf_cuda_h100-2-9-linux.aws.h100_graph63
Pattern hash: e732643dd15b
Shape hash: 095cd511
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([256, 64, 32, 32], bf16))"

class Repro(torch.nn.Module):
    def forward(self, convolution: "bf16[256, 64, 32, 32]"):
        # No stacktrace found for following nodes
        convert_element_type_default: "f32[256, 64, 32, 32]" = torch.ops.prims.convert_element_type.default(convolution, torch.float32);  convolution = None
        gt_scalar: "b8[256, 64, 32, 32]" = torch.ops.aten.gt.Scalar(convert_element_type_default, 0)
        mul_tensor: "f32[256, 64, 32, 32]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 0.2)
        where_self: "f32[256, 64, 32, 32]" = torch.ops.aten.where.self(gt_scalar, convert_element_type_default, mul_tensor);  gt_scalar = convert_element_type_default = mul_tensor = None
        convert_element_type_default_1: "bf16[256, 64, 32, 32]" = torch.ops.prims.convert_element_type.default(where_self, torch.bfloat16);  where_self = None
        return convert_element_type_default_1


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
