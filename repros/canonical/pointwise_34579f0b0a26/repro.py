"""
Standalone repro captured via capture_hook.
Label: torchbench_squeezenet1_1_train
Pattern hash: 34579f0b0a26
Shape hash: e99dcbfa
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
_shapes_config = "(T([512, 1000], f32), T([512, 1000, 13, 13], b8, stride=(169088, 169, 13, 1)), S([512, 1000, 1, 1]), S([512, 1000, 13, 13]))"

class Repro(torch.nn.Module):
    def forward(self, tangents_1: "f32[512, 1000]", le: "b8[512, 1000, 13, 13]", _shape_param_0, _shape_param_1):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:97 in forward, code: return torch.flatten(x, 1)
        reshape_default: "f32[512, 1000, 1, 1]" = torch.ops.aten.reshape.default(tangents_1, _shape_param_0);  tangents_1 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:96 in forward, code: x = self.classifier(x)
        expand_default: "f32[512, 1000, 13, 13]" = torch.ops.aten.expand.default(reshape_default, _shape_param_1);  reshape_default = _shape_param_1 = None
        div_scalar: "f32[512, 1000, 13, 13]" = torch.ops.aten.div.Scalar(expand_default, 169);  expand_default = None
        full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "f32[512, 1000, 13, 13]" = torch.ops.aten.where.self(le, full_default, div_scalar);  le = full_default = div_scalar = None
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
