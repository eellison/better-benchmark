"""
Standalone repro captured via capture_hook.
Label: timm_beit_base_patch16_224_train
Pattern hash: 5f5c323494f9
Shape hash: 6d4a8565
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([25216, 3072], f32), T([25216, 3072], f32), T([3072, 768], f32), S([128, 197, 3072]), S([128, 197, 3072]), S([25216, 3072]))"

class Repro(torch.nn.Module):
    def forward(self, mm_90: "f32[25216, 3072]", addmm_2: "f32[25216, 3072]", primals_19: "f32[3072, 768]", _shape_param_0, _shape_param_1, _shape_param_2):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        reshape_default: "f32[128, 197, 3072]" = torch.ops.aten.reshape.default(mm_90, _shape_param_0);  mm_90 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        reshape_default_1: "f32[128, 197, 3072]" = torch.ops.aten.reshape.default(addmm_2, _shape_param_1);  addmm_2 = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        mul_tensor: "f32[128, 197, 3072]" = torch.ops.aten.mul.Tensor(reshape_default_1, 0.7071067811865476)
        erf_default: "f32[128, 197, 3072]" = torch.ops.aten.erf.default(mul_tensor);  mul_tensor = None
        add_tensor: "f32[128, 197, 3072]" = torch.ops.aten.add.Tensor(erf_default, 1);  erf_default = None
        mul_tensor_1: "f32[128, 197, 3072]" = torch.ops.aten.mul.Tensor(add_tensor, 0.5);  add_tensor = None
        mul_tensor_2: "f32[128, 197, 3072]" = torch.ops.aten.mul.Tensor(reshape_default_1, reshape_default_1)
        mul_tensor_3: "f32[128, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_tensor_2, -0.5);  mul_tensor_2 = None
        exp_default: "f32[128, 197, 3072]" = torch.ops.aten.exp.default(mul_tensor_3);  mul_tensor_3 = None
        mul_tensor_4: "f32[128, 197, 3072]" = torch.ops.aten.mul.Tensor(exp_default, 0.3989422804014327);  exp_default = None
        mul_tensor_5: "f32[128, 197, 3072]" = torch.ops.aten.mul.Tensor(reshape_default_1, mul_tensor_4);  reshape_default_1 = mul_tensor_4 = None
        add_tensor_1: "f32[128, 197, 3072]" = torch.ops.aten.add.Tensor(mul_tensor_1, mul_tensor_5);  mul_tensor_1 = mul_tensor_5 = None
        mul_tensor_6: "f32[128, 197, 3072]" = torch.ops.aten.mul.Tensor(reshape_default, add_tensor_1);  reshape_default = add_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        reshape_default_2: "f32[25216, 3072]" = torch.ops.aten.reshape.default(mul_tensor_6, _shape_param_2);  mul_tensor_6 = _shape_param_2 = None
        permute_default: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_19, [1, 0]);  primals_19 = None
        permute_default_1: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_default, [1, 0]);  permute_default = None
        return (reshape_default_2, permute_default_1)


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
