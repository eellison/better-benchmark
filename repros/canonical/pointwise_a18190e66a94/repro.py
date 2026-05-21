"""
Standalone repro captured via capture_hook.
Label: torchbench_alexnet_train
Pattern hash: a18190e66a94
Shape hash: 6b699954
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
    def forward(self, addmm: "f32[1024, 4096]", inductor_seeds_default: "i64[2]", primals_14: "f32[4096, 4096]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/alexnet.py:51 in forward, code: x = self.classifier(x)
        relu_default: "f32[1024, 4096]" = torch.ops.aten.relu.default(addmm);  addmm = None
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 1);  inductor_seeds_default = None
        inductor_random_default: "f32[1024, 4096]" = torch.ops.prims.inductor_random.default([1024, 4096], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        gt_scalar: "b8[1024, 4096]" = torch.ops.aten.gt.Scalar(inductor_random_default, 0.5);  inductor_random_default = None
        mul_tensor: "f32[1024, 4096]" = torch.ops.aten.mul.Tensor(gt_scalar, relu_default);  gt_scalar = relu_default = None
        mul_tensor_1: "f32[1024, 4096]" = torch.ops.aten.mul.Tensor(mul_tensor, 2.0);  mul_tensor = None
        permute_default: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_14, [1, 0]);  primals_14 = None
        return (mul_tensor_1, permute_default)


def _default_make_inputs():
    return [
    torch.randn([1024, 4096], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [2], dtype=torch.int64, device='cuda'),
    torch.randn([4096, 4096], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
