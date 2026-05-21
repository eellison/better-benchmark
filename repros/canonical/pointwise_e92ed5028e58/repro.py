"""
Standalone repro captured via capture_hook.
Label: torchbench_demucs_train
Pattern hash: e92ed5028e58
Shape hash: 5b5eea9e
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
_shapes_config = "(T([64, 64, 92844], b8, stride=(5947392, 92928, 1)), T([64, 64, 92844], f32))"

class Repro(torch.nn.Module):
    def forward(self, le: "b8[64, 64, 92844]", getitem_3: "f32[64, 64, 92844]"):
        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/demucs/model.py:226 in torch_dynamo_resume_in_forward_at_220, code: x = decode(x)
        full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "f32[64, 64, 92844]" = torch.ops.aten.where.self(le, full_default, getitem_3);  le = full_default = getitem_3 = None
        return where_self



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
