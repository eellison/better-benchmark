"""
Standalone repro captured via capture_hook.
Label: timm_mobilevit_s_infer
Pattern hash: 66a849609a78
Shape hash: 8ab1cb8c
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([8192, 720], f32), S([512, 16, 720]), S([512, 16, 3, 4, 60]))"

class Repro(torch.nn.Module):
    def forward(self, addmm_32: "f32[8192, 720]", _shape_param_0, _shape_param_1):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        reshape_default: "f32[512, 16, 720]" = torch.ops.aten.reshape.default(addmm_32, _shape_param_0);  addmm_32 = _shape_param_0 = None
        reshape_default_1: "f32[512, 16, 3, 4, 60]" = torch.ops.aten.reshape.default(reshape_default, _shape_param_1);  reshape_default = _shape_param_1 = None
        permute_default: "f32[3, 512, 4, 16, 60]" = torch.ops.aten.permute.default(reshape_default_1, [2, 0, 3, 1, 4]);  reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        unbind_int = torch.ops.aten.unbind.int(permute_default);  permute_default = None
        getitem: "f32[512, 4, 16, 60]" = unbind_int[0]
        getitem_1: "f32[512, 4, 16, 60]" = unbind_int[1]
        getitem_2: "f32[512, 4, 16, 60]" = unbind_int[2];  unbind_int = None
        return (getitem, getitem_1, getitem_2)


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
