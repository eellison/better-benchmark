"""
Standalone repro captured via capture_hook.
Label: torchbench_demucs_train
Pattern hash: e6838a9b0d3c
Shape hash: 0f159e25
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
_shapes_config = "(T([64, 8, 371372], f32), T([64, 64, 92844], f32), T([64, 128, 23212], f32), T([64, 256, 5804], f32), T([64, 512, 1452], f32), T([64, 1024, 364], f32), S([64, 4, 2, 371372]))"

class Repro(torch.nn.Module):
    def forward(self, convolution_11: "f32[64, 8, 371372]", relu_4: "f32[64, 64, 92844]", relu_3: "f32[64, 128, 23212]", relu_2: "f32[64, 256, 5804]", relu_1: "f32[64, 512, 1452]", relu: "f32[64, 1024, 364]", _shape_param_0):
        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/demucs/model.py:232 in torch_dynamo_resume_in_forward_at_220, code: x = x.view(x.size(0), self.sources, self.audio_channels, x.size(-1))
        reshape_default: "f32[64, 4, 2, 371372]" = torch.ops.aten.reshape.default(convolution_11, _shape_param_0);  convolution_11 = _shape_param_0 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/demucs/model.py:226 in torch_dynamo_resume_in_forward_at_220, code: x = decode(x)
        le_scalar: "b8[64, 64, 92844]" = torch.ops.aten.le.Scalar(relu_4, 0);  relu_4 = None
        le_scalar_1: "b8[64, 128, 23212]" = torch.ops.aten.le.Scalar(relu_3, 0);  relu_3 = None
        le_scalar_2: "b8[64, 256, 5804]" = torch.ops.aten.le.Scalar(relu_2, 0);  relu_2 = None
        le_scalar_3: "b8[64, 512, 1452]" = torch.ops.aten.le.Scalar(relu_1, 0);  relu_1 = None
        le_scalar_4: "b8[64, 1024, 364]" = torch.ops.aten.le.Scalar(relu, 0);  relu = None
        return (reshape_default, le_scalar, le_scalar_1, le_scalar_2, le_scalar_3, le_scalar_4)



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
