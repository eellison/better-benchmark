"""
Standalone repro captured via capture_hook.
Label: torchbench_alexnet_train
Pattern hash: 2ffb80c8e97c
Shape hash: 53e965f3
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
    def forward(self, addmm_1: "f32[1024, 4096]", primals_16: "f32[1000, 4096]", relu_5: "f32[1024, 4096]", relu_4: "f32[1024, 256, 13, 13]", relu_1: "f32[1024, 192, 27, 27]", relu: "f32[1024, 64, 55, 55]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/alexnet.py:51 in forward, code: x = self.classifier(x)
        relu_default: "f32[1024, 4096]" = torch.ops.aten.relu.default(addmm_1);  addmm_1 = None
        permute_default: "f32[4096, 1000]" = torch.ops.aten.permute.default(primals_16, [1, 0]);  primals_16 = None
        le_scalar: "b8[1024, 4096]" = torch.ops.aten.le.Scalar(relu_5, 0);  relu_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/alexnet.py:48 in forward, code: x = self.features(x)
        le_scalar_1: "b8[1024, 256, 13, 13]" = torch.ops.aten.le.Scalar(relu_4, 0);  relu_4 = None
        le_scalar_2: "b8[1024, 192, 27, 27]" = torch.ops.aten.le.Scalar(relu_1, 0);  relu_1 = None
        le_scalar_3: "b8[1024, 64, 55, 55]" = torch.ops.aten.le.Scalar(relu, 0);  relu = None
        return (relu_default, permute_default, le_scalar, le_scalar_1, le_scalar_2, le_scalar_3)


def _default_make_inputs():
    return [
    torch.randn([1024, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([1000, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 256, 13, 13], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 192, 27, 27], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 64, 55, 55], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
