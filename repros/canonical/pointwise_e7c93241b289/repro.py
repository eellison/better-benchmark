"""
Standalone repro captured via capture_hook.
Label: torchbench_vgg16_train
Pattern hash: e7c93241b289
Shape hash: 5dc3928f
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
    def forward(self, convolution_12: "f32[128, 512, 14, 14]", primals_28: "f32[4096, 25088]", _shape_param_0):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vgg.py:66 in forward, code: x = self.features(x)
        relu_default: "f32[128, 512, 14, 14]" = torch.ops.aten.relu.default(convolution_12);  convolution_12 = None
        _low_memory_max_pool_with_offsets_default = torch.ops.prims._low_memory_max_pool_with_offsets.default(relu_default, [2, 2], [2, 2], [0, 0], [1, 1], False);  relu_default = None
        getitem: "f32[128, 512, 7, 7]" = _low_memory_max_pool_with_offsets_default[0]
        getitem_1: "i8[128, 512, 7, 7]" = _low_memory_max_pool_with_offsets_default[1];  _low_memory_max_pool_with_offsets_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vgg.py:67 in forward, code: x = self.avgpool(x)
        _adaptive_avg_pool2d_default: "f32[128, 512, 7, 7]" = torch.ops.aten._adaptive_avg_pool2d.default(getitem, [7, 7]);  getitem = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vgg.py:68 in forward, code: x = torch.flatten(x, 1)
        reshape_default: "f32[128, 25088]" = torch.ops.aten.reshape.default(_adaptive_avg_pool2d_default, _shape_param_0);  _adaptive_avg_pool2d_default = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vgg.py:69 in forward, code: x = self.classifier(x)
        permute_default: "f32[25088, 4096]" = torch.ops.aten.permute.default(primals_28, [1, 0]);  primals_28 = None
        return (reshape_default, permute_default, getitem_1)


def _default_make_inputs():
    return [
    torch.randn([128, 512, 14, 14], dtype=torch.float32, device='cuda'),
    torch.randn([4096, 25088], dtype=torch.float32, device='cuda'),
    [128, 25088],  # _shape_param_0
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
