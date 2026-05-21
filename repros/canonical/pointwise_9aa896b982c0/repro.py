"""
Standalone repro captured via capture_hook.
Label: torchbench_alexnet_infer
Pattern hash: 9aa896b982c0
Shape hash: 55481bb0
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
    def forward(self, convolution_4: "f16[1024, 256, 13, 13]", arg11_1: "f16[4096, 9216]", _shape_param_0):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/alexnet.py:48 in forward, code: x = self.features(x)
        relu_default: "f16[1024, 256, 13, 13]" = torch.ops.aten.relu.default(convolution_4);  convolution_4 = None
        _low_memory_max_pool_with_offsets_default = torch.ops.prims._low_memory_max_pool_with_offsets.default(relu_default, [3, 3], [2, 2], [0, 0], [1, 1], False);  relu_default = None
        getitem: "f16[1024, 256, 6, 6]" = _low_memory_max_pool_with_offsets_default[0];  _low_memory_max_pool_with_offsets_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/alexnet.py:49 in forward, code: x = self.avgpool(x)
        _adaptive_avg_pool2d_default: "f16[1024, 256, 6, 6]" = torch.ops.aten._adaptive_avg_pool2d.default(getitem, [6, 6]);  getitem = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/alexnet.py:50 in forward, code: x = torch.flatten(x, 1)
        reshape_default: "f16[1024, 9216]" = torch.ops.aten.reshape.default(_adaptive_avg_pool2d_default, _shape_param_0);  _adaptive_avg_pool2d_default = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/alexnet.py:51 in forward, code: x = self.classifier(x)
        permute_default: "f16[9216, 4096]" = torch.ops.aten.permute.default(arg11_1, [1, 0]);  arg11_1 = None
        return (reshape_default, permute_default)


def _default_make_inputs():
    return [
    torch.randn([1024, 256, 13, 13], dtype=torch.float16, device='cuda'),
    torch.randn([4096, 9216], dtype=torch.float16, device='cuda'),
    [1024, 9216],  # _shape_param_0
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
