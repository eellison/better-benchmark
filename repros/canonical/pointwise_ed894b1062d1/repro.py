"""
Standalone repro captured via capture_hook.
Label: timm_beit_base_patch16_224_infer
Pattern hash: ed894b1062d1
Shape hash: 5a1fbf8b
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
_shapes_config = "(T([25216, 2304], f32), S([128, 197, 2304]), S([128, 197, 3, 12, -1]))"

class Repro(torch.nn.Module):
    def forward(self, addmm_36: "f32[25216, 2304]", _shape_param_0, _shape_param_1):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:218 in forward, code: qkv = F.linear(x, weight=self.qkv.weight, bias=qkv_bias)
        reshape_default: "f32[128, 197, 2304]" = torch.ops.aten.reshape.default(addmm_36, _shape_param_0);  addmm_36 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:219 in forward, code: qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        reshape_default_1: "f32[128, 197, 3, 12, 64]" = torch.ops.aten.reshape.default(reshape_default, _shape_param_1);  reshape_default = _shape_param_1 = None
        permute_default: "f32[3, 128, 12, 197, 64]" = torch.ops.aten.permute.default(reshape_default_1, [2, 0, 3, 1, 4]);  reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:220 in forward, code: q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim
        unbind_int = torch.ops.aten.unbind.int(permute_default);  permute_default = None
        getitem: "f32[128, 12, 197, 64]" = unbind_int[0]
        getitem_1: "f32[128, 12, 197, 64]" = unbind_int[1]
        getitem_2: "f32[128, 12, 197, 64]" = unbind_int[2];  unbind_int = None
        return (getitem_1, getitem_2, getitem)



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
