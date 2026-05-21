"""
Standalone repro captured via capture_hook.
Label: torchbench_squeezenet1_1_train
Pattern hash: aee060b2e179
Shape hash: ed34fe42
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, getitem_69: "f32[512, 128, 55, 55]", le_22: "b8[512, 64, 55, 55]", full_default: "f32[]", le_23: "b8[512, 64, 55, 55]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:31 in forward, code: return torch.cat(
        slice_tensor: "f32[512, 64, 55, 55]" = torch.ops.aten.slice.Tensor(getitem_69, 1, 0, 64)
        slice_tensor_1: "f32[512, 64, 55, 55]" = torch.ops.aten.slice.Tensor(getitem_69, 1, 64, 128);  getitem_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:32 in forward, code: [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
        where_self: "f32[512, 64, 55, 55]" = torch.ops.aten.where.self(le_22, full_default, slice_tensor_1);  le_22 = slice_tensor_1 = None
        where_self_1: "f32[512, 64, 55, 55]" = torch.ops.aten.where.self(le_23, full_default, slice_tensor);  le_23 = full_default = slice_tensor = None
        return (where_self, where_self_1)


def _default_make_inputs():
    return [
    torch.randn(198246400, dtype=torch.float32, device='cuda').as_strided([512, 128, 55, 55], [387200, 1, 7040, 128]),  # getitem_69
    torch.randint(0, 2, (99123200,), dtype=torch.bool, device='cuda').as_strided([512, 64, 55, 55], [193600, 1, 3520, 64]),  # le_22
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, (99123200,), dtype=torch.bool, device='cuda').as_strided([512, 64, 55, 55], [193600, 1, 3520, 64]),  # le_23
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
