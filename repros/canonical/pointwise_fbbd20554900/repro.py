"""
Standalone repro captured via capture_hook.
Label: vgg16_training
Pattern hash: fbbd20554900
Shape hash: e6c29dc6
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_prelude import *  # noqa: F401,F403
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, addmm_1: "f32[4, 4096]", inductor_seeds_default: "i64[2]", primals_32: "f32[1000, 4096]", relu_13: "f32[4, 4096]", relu_12: "f32[4, 512, 14, 14]", relu_9: "f32[4, 512, 28, 28]", relu_6: "f32[4, 256, 56, 56]", relu_3: "f32[4, 128, 112, 112]", relu_1: "f32[4, 64, 224, 224]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vgg.py:69 in forward, code: x = self.classifier(x)
        relu_default: "f32[4, 4096]" = torch.ops.aten.relu.default(addmm_1);  addmm_1 = None
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 1);  inductor_seeds_default = None
        inductor_random_default: "f32[4, 4096]" = torch.ops.prims.inductor_random.default([4, 4096], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        gt_scalar: "b8[4, 4096]" = torch.ops.aten.gt.Scalar(inductor_random_default, 0.5);  inductor_random_default = None
        mul_tensor: "f32[4, 4096]" = torch.ops.aten.mul.Tensor(gt_scalar, relu_default);  gt_scalar = None
        mul_tensor_1: "f32[4, 4096]" = torch.ops.aten.mul.Tensor(mul_tensor, 2.0);  mul_tensor = None
        permute_default: "f32[4096, 1000]" = torch.ops.aten.permute.default(primals_32, [1, 0]);  primals_32 = None
        le_scalar: "b8[4, 4096]" = torch.ops.aten.le.Scalar(relu_default, 0);  relu_default = None
        le_scalar_1: "b8[4, 4096]" = torch.ops.aten.le.Scalar(relu_13, 0);  relu_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vgg.py:66 in forward, code: x = self.features(x)
        le_scalar_2: "b8[4, 512, 14, 14]" = torch.ops.aten.le.Scalar(relu_12, 0);  relu_12 = None
        le_scalar_3: "b8[4, 512, 28, 28]" = torch.ops.aten.le.Scalar(relu_9, 0);  relu_9 = None
        le_scalar_4: "b8[4, 256, 56, 56]" = torch.ops.aten.le.Scalar(relu_6, 0);  relu_6 = None
        le_scalar_5: "b8[4, 128, 112, 112]" = torch.ops.aten.le.Scalar(relu_3, 0);  relu_3 = None
        le_scalar_6: "b8[4, 64, 224, 224]" = torch.ops.aten.le.Scalar(relu_1, 0);  relu_1 = None
        return (mul_tensor_1, permute_default, le_scalar, le_scalar_1, le_scalar_2, le_scalar_3, le_scalar_4, le_scalar_5, le_scalar_6)


def _default_make_inputs():
    return [
    torch.randn([4, 4096], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [2], dtype=torch.int64, device='cuda'),
    torch.randn([1000, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([4, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 14, 14], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 28, 28], dtype=torch.float32, device='cuda'),
    torch.randn([4, 256, 56, 56], dtype=torch.float32, device='cuda'),
    torch.randn([4, 128, 112, 112], dtype=torch.float32, device='cuda'),
    torch.randn([4, 64, 224, 224], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
