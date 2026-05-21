"""
Standalone repro captured via capture_hook.
Label: timm_vit_base_patch16_siglip_256_infer
Pattern hash: 4ebc1e147929
Shape hash: b742dacb
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
_shapes_config = "(T([128, 768], f32), T([128, 1, 768], f32), S([128, 1, 768]))"

class Repro(torch.nn.Module):
    def forward(self, addmm_51: "f32[128, 768]", view_129: "f32[128, 1, 768]", _shape_param_0):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        reshape_default: "f32[128, 1, 768]" = torch.ops.aten.reshape.default(addmm_51, _shape_param_0);  addmm_51 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention_pool.py:122 in forward, code: x = x + self.mlp(self.norm(x))
        add_tensor: "f32[128, 1, 768]" = torch.ops.aten.add.Tensor(view_129, reshape_default);  view_129 = reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention_pool.py:126 in forward, code: x = x[:, 0]
        select_int: "f32[128, 768]" = torch.ops.aten.select.int(add_tensor, 1, 0);  add_tensor = None
        return select_int



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
