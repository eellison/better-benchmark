"""
Standalone repro captured via capture_hook.
Label: torchbench_squeezenet1_1_train
Pattern hash: c7489acdbe1e
Shape hash: 794de873
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
_shapes_config = "(T([512, 256, 13, 13], f32), T([512, 256, 13, 13], f32))"

class Repro(torch.nn.Module):
    def forward(self, convolution_23: "f32[512, 256, 13, 13]", convolution_24: "f32[512, 256, 13, 13]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:32 in forward, code: [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
        relu_default: "f32[512, 256, 13, 13]" = torch.ops.aten.relu.default(convolution_23);  convolution_23 = None
        relu_default_1: "f32[512, 256, 13, 13]" = torch.ops.aten.relu.default(convolution_24);  convolution_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:31 in forward, code: return torch.cat(
        cat_default: "f32[512, 512, 13, 13]" = torch.ops.aten.cat.default([relu_default, relu_default_1], 1);  relu_default = relu_default_1 = None

        # No stacktrace found for following nodes
        inductor_seeds_default: "i64[1]" = torch.ops.prims.inductor_seeds.default(1, device(type='cuda', index=0))

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:96 in forward, code: x = self.classifier(x)
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 0);  inductor_seeds_default = None
        inductor_random_default: "f32[512, 512, 13, 13]" = torch.ops.prims.inductor_random.default([512, 512, 13, 13], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        gt_scalar: "b8[512, 512, 13, 13]" = torch.ops.aten.gt.Scalar(inductor_random_default, 0.5);  inductor_random_default = None
        mul_tensor: "f32[512, 512, 13, 13]" = torch.ops.aten.mul.Tensor(gt_scalar, cat_default);  gt_scalar = cat_default = None
        mul_tensor_1: "f32[512, 512, 13, 13]" = torch.ops.aten.mul.Tensor(mul_tensor, 2.0);  mul_tensor = None
        return mul_tensor_1



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
