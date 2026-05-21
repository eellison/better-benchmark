"""
Standalone repro captured via capture_hook.
Label: torchbench_squeezenet1_1_train
Pattern hash: 5bb978a02297
Shape hash: 98c2700a
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
    def forward(self, getitem_72: "f32[512, 16, 55, 55]", getitem_75: "f32[512, 16, 55, 55]", relu_1: "f32[512, 16, 55, 55]", full_default: "f32[]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:32 in forward, code: [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
        add_tensor: "f32[512, 16, 55, 55]" = torch.ops.aten.add.Tensor(getitem_72, getitem_75);  getitem_72 = getitem_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:30 in forward, code: x = self.squeeze_activation(self.squeeze(x))
        le_scalar: "b8[512, 16, 55, 55]" = torch.ops.aten.le.Scalar(relu_1, 0);  relu_1 = None
        where_self: "f32[512, 16, 55, 55]" = torch.ops.aten.where.self(le_scalar, full_default, add_tensor);  le_scalar = full_default = add_tensor = None
        return where_self


def _default_make_inputs():
    return [
    torch.randn(24780800, dtype=torch.float32, device='cuda').as_strided([512, 16, 55, 55], [48400, 1, 880, 16]),  # getitem_72
    torch.randn(24780800, dtype=torch.float32, device='cuda').as_strided([512, 16, 55, 55], [48400, 1, 880, 16]),  # getitem_75
    torch.randn(24780800, dtype=torch.float32, device='cuda').as_strided([512, 16, 55, 55], [48400, 1, 880, 16]),  # relu_1
    torch.randn([], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
