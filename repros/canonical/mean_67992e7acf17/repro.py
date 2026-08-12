"""
Standalone repro captured via capture_hook.
Label: torchbench_opacus_cifar10_infer
Pattern hash: 67992e7acf17
Shape hash: a7a01247
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 3
# Input shapes/strides/dtypes live in the sibling shapes.json (structured,
# one entry per point); forward()'s annotations document the default shapes
# inline. Default inputs = the first shapes.json point.

class Repro(torch.nn.Module):
    def forward(self, arg1_1: "f32[64, 512, 1, 1]", arg0_1: "Sym(512)"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:278 in torch_dynamo_resume_in__forward_impl_at_276, code: x = self.avgpool(x)
        mean_dim: "f32[64, 512, 1, 1]" = torch.ops.aten.mean.dim(arg1_1, [-1, -2], True);  arg1_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:279 in torch_dynamo_resume_in__forward_impl_at_276, code: x = torch.flatten(x, 1)
        reshape_default: "f32[64, 512]" = torch.ops.aten.reshape.default(mean_dim, [64, arg0_1]);  mean_dim = arg0_1 = None
        return reshape_default



def _default_make_inputs():
    configs = load_shape_configs(__file__)
    if not configs:
        raise RuntimeError(
            "no shapes.json next to this repro — pass an explicit config "
            "via make_inputs(shape_config=...)")
    return make_inputs_from_config(next(iter(configs.values())))


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
