"""
Standalone repro captured via capture_hook.
Label: torchbench_opacus_cifar10_infer
Pattern hash: 923310333f84
Shape hash: 36a5d0e0
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
_shapes_config = "(T([64, 512, 1, 1], f32), S([512]), S([64, arg0_1]))"

class Repro(torch.nn.Module):
    def forward(self, arg1_1: "f32[64, 512, 1, 1]", arg0_1: "Sym(512)", _shape_param_0):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:278 in torch_dynamo_resume_in__forward_impl_at_276, code: x = self.avgpool(x)
        mean_dim: "f32[64, 512, 1, 1]" = torch.ops.aten.mean.dim(arg1_1, [-1, -2], True);  arg1_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:279 in torch_dynamo_resume_in__forward_impl_at_276, code: x = torch.flatten(x, 1)
        reshape_default: "f32[64, 512]" = torch.ops.aten.reshape.default(mean_dim, _shape_param_0);  mean_dim = _shape_param_0 = None
        return reshape_default



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
