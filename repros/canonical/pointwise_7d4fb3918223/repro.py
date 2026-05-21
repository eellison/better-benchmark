"""
Standalone repro captured via capture_hook.
Label: torchbench_alexnet_train
Pattern hash: 7d4fb3918223
Shape hash: 84e52a3a
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
_shapes_config = "(T([1024, 4096], f32), T([1024, 4096], f32), T([4096, 4096], f32))"

class Repro(torch.nn.Module):
    def forward(self, relu_6: "f32[1024, 4096]", mm: "f32[1024, 4096]", primals_14: "f32[4096, 4096]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/alexnet.py:51 in forward, code: x = self.classifier(x)
        le_scalar: "b8[1024, 4096]" = torch.ops.aten.le.Scalar(relu_6, 0);  relu_6 = None
        full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "f32[1024, 4096]" = torch.ops.aten.where.self(le_scalar, full_default, mm);  le_scalar = full_default = mm = None
        permute_default: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_14, [1, 0]);  primals_14 = None
        permute_default_1: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_default, [1, 0]);  permute_default = None
        return (where_self, permute_default_1)



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
