"""
Standalone repro captured via capture_hook.
Label: timm_mobilevit_s_infer
Pattern hash: eecc7f6f5a71
Shape hash: 83d1fcdd
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([8192, 480], f32), T([240, 480], f32), S([512, 16, 480]), S([8192, 480]))"

class Repro(torch.nn.Module):
    def forward(self, addmm_34: "f32[8192, 480]", arg259_1: "f32[240, 480]", _shape_param_0, _shape_param_1):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        reshape_default: "f32[512, 16, 480]" = torch.ops.aten.reshape.default(addmm_34, _shape_param_0);  addmm_34 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        neg_default: "f32[512, 16, 480]" = torch.ops.aten.neg.default(reshape_default)
        exp_default: "f32[512, 16, 480]" = torch.ops.aten.exp.default(neg_default);  neg_default = None
        add_tensor: "f32[512, 16, 480]" = torch.ops.aten.add.Tensor(exp_default, 1);  exp_default = None
        div_tensor: "f32[512, 16, 480]" = torch.ops.aten.div.Tensor(reshape_default, add_tensor);  reshape_default = add_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        reshape_default_1: "f32[8192, 480]" = torch.ops.aten.reshape.default(div_tensor, _shape_param_1);  div_tensor = _shape_param_1 = None
        permute_default: "f32[480, 240]" = torch.ops.aten.permute.default(arg259_1, [1, 0]);  arg259_1 = None
        return (reshape_default_1, permute_default)


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
