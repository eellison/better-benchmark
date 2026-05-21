"""
Standalone repro captured via capture_hook.
Label: torchbench_modded_nanogpt_infer
Pattern hash: f802213a6120
Shape hash: 3950c601
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
_shapes_config = "(T([6144, 3072], bf16), S([1, 6144, 3072]), S([6144, 3072]))"

class Repro(torch.nn.Module):
    def forward(self, mm_44: "bf16[6144, 3072]", _shape_param_0, _shape_param_1):
        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:723 in forward, code: x = F.linear(x, self.c_fc.T.type_as(x))
        reshape_default: "bf16[1, 6144, 3072]" = torch.ops.aten.reshape.default(mm_44, _shape_param_0);  mm_44 = _shape_param_0 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:724 in forward, code: x = F.relu(
        relu_default: "bf16[1, 6144, 3072]" = torch.ops.aten.relu.default(reshape_default);  reshape_default = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:726 in forward, code: ).square()  # https://arxiv.org/abs/2109.08668v2; ~1-2% better than GELU; suggested by @SKYLINEZ007 and @Grad62304977
        pow_tensor_scalar: "bf16[1, 6144, 3072]" = torch.ops.aten.pow.Tensor_Scalar(relu_default, 2);  relu_default = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:727 in forward, code: x = F.linear(x, self.c_proj.type_as(x))
        reshape_default_1: "bf16[6144, 3072]" = torch.ops.aten.reshape.default(pow_tensor_scalar, _shape_param_1);  pow_tensor_scalar = _shape_param_1 = None
        return reshape_default_1



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
