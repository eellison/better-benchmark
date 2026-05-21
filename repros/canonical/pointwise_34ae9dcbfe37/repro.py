"""
Standalone repro captured via capture_hook.
Label: torchbench_squeezenet1_1_train
Pattern hash: 34ae9dcbfe37
Shape hash: 6d2d0d28
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
_shapes_config = "(T([512, 128, 55, 55], f32, stride=(387200, 1, 7040, 128)), T([512, 64, 55, 55], b8, stride=(193600, 1, 3520, 64)), T([], f32))"

class Repro(torch.nn.Module):
    def forward(self, getitem_69: "f32[512, 128, 55, 55]", le_23: "b8[512, 64, 55, 55]", full_default: "f32[]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:31 in forward, code: return torch.cat(
        slice_tensor: "f32[512, 64, 55, 55]" = torch.ops.aten.slice.Tensor(getitem_69, 1, 0, 64);  getitem_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:32 in forward, code: [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
        where_self: "f32[512, 64, 55, 55]" = torch.ops.aten.where.self(le_23, full_default, slice_tensor);  le_23 = full_default = slice_tensor = None
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
