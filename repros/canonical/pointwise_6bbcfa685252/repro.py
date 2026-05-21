"""
Standalone repro captured via capture_hook.
Label: torchbench_alexnet_train
Pattern hash: 6bbcfa685252
Shape hash: 8f38ef61
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
    def forward(self, gt: "b8[1024, 9216]", mm_4: "f32[1024, 9216]", _shape_param_0):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/alexnet.py:51 in forward, code: x = self.classifier(x)
        convert_element_type_default: "f32[1024, 9216]" = torch.ops.prims.convert_element_type.default(gt, torch.float32);  gt = None
        mul_tensor: "f32[1024, 9216]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 2.0);  convert_element_type_default = None
        mul_tensor_1: "f32[1024, 9216]" = torch.ops.aten.mul.Tensor(mm_4, mul_tensor);  mm_4 = mul_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/alexnet.py:50 in forward, code: x = torch.flatten(x, 1)
        reshape_default: "f32[1024, 256, 6, 6]" = torch.ops.aten.reshape.default(mul_tensor_1, _shape_param_0);  mul_tensor_1 = _shape_param_0 = None
        return reshape_default


def _default_make_inputs():
    return [
    torch.randint(0, 2, [1024, 9216], dtype=torch.bool, device='cuda'),
    torch.randn([1024, 9216], dtype=torch.float32, device='cuda'),
    [1024, 256, 6, 6],  # _shape_param_0
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
