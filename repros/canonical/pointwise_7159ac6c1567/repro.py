"""
Standalone repro captured via capture_hook.
Label: torchbench_dlrm_train
Pattern hash: 7159ac6c1567
Shape hash: 2de803a0
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
_shapes_config = "(T([2048, 1], b8), T([2048, 1], f32), T([1, 1024], f32), T([2048, 1024], f32))"

class Repro(torch.nn.Module):
    def forward(self, le: "b8[2048, 1]", tangents_1: "f32[2048, 1]", primals_29: "f32[1, 1024]", relu_4: "f32[2048, 1024]"):
        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/dlrm/dlrm_s_pytorch.py:284 in apply_mlp, code: return layers(x)
        full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "f32[2048, 1]" = torch.ops.aten.where.self(le, full_default, tangents_1);  le = tangents_1 = None
        permute_default: "f32[1024, 1]" = torch.ops.aten.permute.default(primals_29, [1, 0]);  primals_29 = None
        permute_default_1: "f32[1, 1024]" = torch.ops.aten.permute.default(permute_default, [1, 0]);  permute_default = None
        mul_tensor: "f32[2048, 1024]" = torch.ops.aten.mul.Tensor(where_self, permute_default_1);  where_self = permute_default_1 = None
        le_scalar: "b8[2048, 1024]" = torch.ops.aten.le.Scalar(relu_4, 0);  relu_4 = None
        where_self_1: "f32[2048, 1024]" = torch.ops.aten.where.self(le_scalar, full_default, mul_tensor);  le_scalar = full_default = mul_tensor = None
        return where_self_1



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
