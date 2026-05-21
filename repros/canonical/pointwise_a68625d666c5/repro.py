"""
Standalone repro captured via capture_hook.
Label: torchbench_opacus_cifar10_infer
Pattern hash: a68625d666c5
Shape hash: 9d8dea96
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
_shapes_config = "(T([64, 64, 16, 16], f32))"

class Repro(torch.nn.Module):
    def forward(self, arg0_1: "f32[64, 64, 16, 16]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:270 in torch_dynamo_resume_in__forward_impl_at_269, code: x = self.relu(x)
        relu_default: "f32[64, 64, 16, 16]" = torch.ops.aten.relu.default(arg0_1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:271 in torch_dynamo_resume_in__forward_impl_at_269, code: x = self.maxpool(x)
        _low_memory_max_pool_with_offsets_default = torch.ops.prims._low_memory_max_pool_with_offsets.default(relu_default, [3, 3], [2, 2], [1, 1], [1, 1], False)
        getitem: "f32[64, 64, 8, 8]" = _low_memory_max_pool_with_offsets_default[0];  _low_memory_max_pool_with_offsets_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:270 in torch_dynamo_resume_in__forward_impl_at_269, code: x = self.relu(x)
        copy__default: "f32[64, 64, 16, 16]" = torch.ops.aten.copy_.default(arg0_1, relu_default);  arg0_1 = relu_default = None
        return (copy__default, getitem)



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
