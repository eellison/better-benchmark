"""
Standalone repro captured via capture_hook.
Label: torchbench_alexnet_infer
Pattern hash: 1e5ed2d9dbaf
Shape hash: 88955a70
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
_shapes_config = "(T([1024, 4096], f16), T([1000, 4096], f16))"

class Repro(torch.nn.Module):
    def forward(self, addmm_1: "f16[1024, 4096]", arg15_1: "f16[1000, 4096]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/alexnet.py:51 in forward, code: x = self.classifier(x)
        relu_default: "f16[1024, 4096]" = torch.ops.aten.relu.default(addmm_1);  addmm_1 = None
        permute_default: "f16[4096, 1000]" = torch.ops.aten.permute.default(arg15_1, [1, 0]);  arg15_1 = None
        return (relu_default, permute_default)



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
