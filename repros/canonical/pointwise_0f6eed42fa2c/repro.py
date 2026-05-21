"""
Standalone repro captured via capture_hook.
Label: timm_swin_base_patch4_window7_224_infer
Pattern hash: 0f6eed42fa2c
Shape hash: 77bc7cfa
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
_shapes_config = "(T([4096, 49, 32], f32), S([128, 32, 49, 32]), S([128, 49, 1024]), S([6272, 1024]))"

class Repro(torch.nn.Module):
    def forward(self, bmm_47: "f32[4096, 49, 32]", _shape_param_0, _shape_param_1, _shape_param_2):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        reshape_default: "f32[128, 32, 49, 32]" = torch.ops.aten.reshape.default(bmm_47, _shape_param_0);  bmm_47 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:245 in forward, code: x = x.transpose(1, 2).reshape(B_, N, -1)
        permute_default: "f32[128, 49, 32, 32]" = torch.ops.aten.permute.default(reshape_default, [0, 2, 1, 3]);  reshape_default = None
        clone_default: "f32[128, 49, 32, 32]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        reshape_default_1: "f32[128, 49, 1024]" = torch.ops.aten.reshape.default(clone_default, _shape_param_1);  clone_default = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        reshape_default_2: "f32[6272, 1024]" = torch.ops.aten.reshape.default(reshape_default_1, _shape_param_2);  reshape_default_1 = _shape_param_2 = None
        return reshape_default_2



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
