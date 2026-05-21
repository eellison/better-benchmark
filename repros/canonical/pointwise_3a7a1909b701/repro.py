"""
Standalone repro captured via capture_hook.
Label: torchbench_alexnet_train
Pattern hash: 3a7a1909b701
Shape hash: 6438b092
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
    def forward(self, convolution_4: "f32[1024, 256, 13, 13]", primals_12: "f32[4096, 9216]", _shape_param_0):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/alexnet.py:48 in forward, code: x = self.features(x)
        relu_default: "f32[1024, 256, 13, 13]" = torch.ops.aten.relu.default(convolution_4);  convolution_4 = None
        _low_memory_max_pool_with_offsets_default = torch.ops.prims._low_memory_max_pool_with_offsets.default(relu_default, [3, 3], [2, 2], [0, 0], [1, 1], False);  relu_default = None
        getitem: "f32[1024, 256, 6, 6]" = _low_memory_max_pool_with_offsets_default[0]
        getitem_1: "i8[1024, 256, 6, 6]" = _low_memory_max_pool_with_offsets_default[1];  _low_memory_max_pool_with_offsets_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/alexnet.py:49 in forward, code: x = self.avgpool(x)
        _adaptive_avg_pool2d_default: "f32[1024, 256, 6, 6]" = torch.ops.aten._adaptive_avg_pool2d.default(getitem, [6, 6]);  getitem = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/alexnet.py:50 in forward, code: x = torch.flatten(x, 1)
        reshape_default: "f32[1024, 9216]" = torch.ops.aten.reshape.default(_adaptive_avg_pool2d_default, _shape_param_0);  _adaptive_avg_pool2d_default = _shape_param_0 = None

        # No stacktrace found for following nodes
        inductor_seeds_default: "i64[2]" = torch.ops.prims.inductor_seeds.default(2, device(type='cuda', index=0))

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/alexnet.py:51 in forward, code: x = self.classifier(x)
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 0);  inductor_seeds_default = None
        inductor_random_default: "f32[1024, 9216]" = torch.ops.prims.inductor_random.default([1024, 9216], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        gt_scalar: "b8[1024, 9216]" = torch.ops.aten.gt.Scalar(inductor_random_default, 0.5);  inductor_random_default = None
        mul_tensor: "f32[1024, 9216]" = torch.ops.aten.mul.Tensor(gt_scalar, reshape_default);  gt_scalar = reshape_default = None
        mul_tensor_1: "f32[1024, 9216]" = torch.ops.aten.mul.Tensor(mul_tensor, 2.0);  mul_tensor = None
        permute_default: "f32[9216, 4096]" = torch.ops.aten.permute.default(primals_12, [1, 0]);  primals_12 = None
        return (mul_tensor_1, permute_default, getitem_1)


def _default_make_inputs():
    return [
    torch.randn([1024, 256, 13, 13], dtype=torch.float32, device='cuda'),
    torch.randn([4096, 9216], dtype=torch.float32, device='cuda'),
    [1024, 9216],  # _shape_param_0
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
