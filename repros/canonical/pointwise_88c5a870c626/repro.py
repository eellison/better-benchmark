"""
Standalone repro captured via capture_hook.
Label: torchbench_dlrm_train
Pattern hash: 88c5a870c626
Shape hash: 216a2c8f
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
_shapes_config = "(T([2048, 100], f32), T([2048, 64, 9], f32), T([2048, 9, 64], f32), T([2048, 64], b8), T([], f32), S([2048, 576]))"

class Repro(torch.nn.Module):
    def forward(self, mm_5: "f32[2048, 100]", bmm_1: "f32[2048, 64, 9]", bmm_2: "f32[2048, 9, 64]", le_4: "b8[2048, 64]", full_default: "f32[]", _shape_param_0):
        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/dlrm/dlrm_s_pytorch.py:337 in interact_features, code: R = torch.cat([x] + [Zflat], dim=1)
        slice_tensor: "f32[2048, 64]" = torch.ops.aten.slice.Tensor(mm_5, 1, 0, 64);  mm_5 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/dlrm/dlrm_s_pytorch.py:318 in interact_features, code: Z = torch.bmm(T, torch.transpose(T, 1, 2))
        permute_default: "f32[2048, 9, 64]" = torch.ops.aten.permute.default(bmm_1, [0, 2, 1]);  bmm_1 = None
        add_tensor: "f32[2048, 9, 64]" = torch.ops.aten.add.Tensor(bmm_2, permute_default);  bmm_2 = permute_default = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/dlrm/dlrm_s_pytorch.py:316 in interact_features, code: T = torch.cat([x] + ly, dim=1).view((batch_size, -1, d))
        reshape_default: "f32[2048, 576]" = torch.ops.aten.reshape.default(add_tensor, _shape_param_0);  add_tensor = _shape_param_0 = None
        slice_tensor_1: "f32[2048, 64]" = torch.ops.aten.slice.Tensor(reshape_default, 1, 0, 64);  reshape_default = None
        add_tensor_1: "f32[2048, 64]" = torch.ops.aten.add.Tensor(slice_tensor, slice_tensor_1);  slice_tensor = slice_tensor_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/dlrm/dlrm_s_pytorch.py:284 in apply_mlp, code: return layers(x)
        where_self: "f32[2048, 64]" = torch.ops.aten.where.self(le_4, full_default, add_tensor_1);  le_4 = full_default = add_tensor_1 = None
        return where_self



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
