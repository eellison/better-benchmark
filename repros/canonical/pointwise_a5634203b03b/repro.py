"""
Standalone repro captured via capture_hook.
Label: torchbench_demucs_infer
Pattern hash: a5634203b03b
Shape hash: b996b6bc
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
_shapes_config = "(T([8, 64, 92844], f32), T([8, 64, 95696], f32))"

class Repro(torch.nn.Module):
    def forward(self, convolution_9: "f32[8, 64, 92844]", arg26_1: "f32[8, 64, 95696]"):
        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/demucs/model.py:226 in torch_dynamo_resume_in_forward_at_220, code: x = decode(x)
        relu_default: "f32[8, 64, 92844]" = torch.ops.aten.relu.default(convolution_9);  convolution_9 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/demucs/utils.py:35 in center_trim, code: tensor = tensor[..., delta // 2 : -(delta - delta // 2)]
        slice_tensor: "f32[8, 64, 92844]" = torch.ops.aten.slice.Tensor(arg26_1, 2, 1426, -1426);  arg26_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/demucs/model.py:225 in torch_dynamo_resume_in_forward_at_220, code: x = x + skip
        add_tensor: "f32[8, 64, 92844]" = torch.ops.aten.add.Tensor(relu_default, slice_tensor);  relu_default = slice_tensor = None
        return add_tensor



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
