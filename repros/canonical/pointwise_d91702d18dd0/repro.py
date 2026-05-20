"""
Standalone repro captured via capture_hook.
Label: timm_beit_base_patch16_224_infer
Pattern hash: d91702d18dd0
Shape hash: 62ff1f5d
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([25216, 3072], f32), T([768, 3072], f32), S([128, 197, 3072]), S([25216, 3072]))"

class Repro(torch.nn.Module):
    def forward(self, addmm_46: "f32[25216, 3072]", arg218_1: "f32[768, 3072]", _shape_param_0, _shape_param_1):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        reshape_default: "f32[128, 197, 3072]" = torch.ops.aten.reshape.default(addmm_46, _shape_param_0);  addmm_46 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        mul_tensor: "f32[128, 197, 3072]" = torch.ops.aten.mul.Tensor(reshape_default, 0.5)
        mul_tensor_1: "f32[128, 197, 3072]" = torch.ops.aten.mul.Tensor(reshape_default, 0.7071067811865476);  reshape_default = None
        erf_default: "f32[128, 197, 3072]" = torch.ops.aten.erf.default(mul_tensor_1);  mul_tensor_1 = None
        add_tensor: "f32[128, 197, 3072]" = torch.ops.aten.add.Tensor(erf_default, 1);  erf_default = None
        mul_tensor_2: "f32[128, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_tensor, add_tensor);  mul_tensor = add_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        reshape_default_1: "f32[25216, 3072]" = torch.ops.aten.reshape.default(mul_tensor_2, _shape_param_1);  mul_tensor_2 = _shape_param_1 = None
        permute_default: "f32[3072, 768]" = torch.ops.aten.permute.default(arg218_1, [1, 0]);  arg218_1 = None
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
