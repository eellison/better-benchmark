"""
Standalone repro captured via capture_hook.
Label: timm_mobilevit_s_train
Pattern hash: e0895d2cdf49
Shape hash: af0f7478
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, mm_66: "f32[131072, 288]", addmm_2: "f32[131072, 288]", primals_113: "f32[288, 144]", _shape_param_0, _shape_param_1, _shape_param_2):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        reshape_default: "f32[512, 256, 288]" = torch.ops.aten.reshape.default(mm_66, _shape_param_0);  mm_66 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        reshape_default_1: "f32[512, 256, 288]" = torch.ops.aten.reshape.default(addmm_2, _shape_param_1);  addmm_2 = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        neg_default: "f32[512, 256, 288]" = torch.ops.aten.neg.default(reshape_default_1)
        exp_default: "f32[512, 256, 288]" = torch.ops.aten.exp.default(neg_default);  neg_default = None
        add_tensor: "f32[512, 256, 288]" = torch.ops.aten.add.Tensor(exp_default, 1);  exp_default = None
        reciprocal_default: "f32[512, 256, 288]" = torch.ops.aten.reciprocal.default(add_tensor);  add_tensor = None
        mul_tensor: "f32[512, 256, 288]" = torch.ops.aten.mul.Tensor(reciprocal_default, 1);  reciprocal_default = None
        mul_tensor_1: "f32[512, 256, 288]" = torch.ops.aten.mul.Tensor(reshape_default, mul_tensor);  reshape_default = None
        sub_tensor: "f32[512, 256, 288]" = torch.ops.aten.sub.Tensor(1, mul_tensor);  mul_tensor = None
        mul_tensor_2: "f32[512, 256, 288]" = torch.ops.aten.mul.Tensor(reshape_default_1, sub_tensor);  reshape_default_1 = sub_tensor = None
        add_tensor_1: "f32[512, 256, 288]" = torch.ops.aten.add.Tensor(mul_tensor_2, 1);  mul_tensor_2 = None
        mul_tensor_3: "f32[512, 256, 288]" = torch.ops.aten.mul.Tensor(mul_tensor_1, add_tensor_1);  mul_tensor_1 = add_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        reshape_default_2: "f32[131072, 288]" = torch.ops.aten.reshape.default(mul_tensor_3, _shape_param_2);  mul_tensor_3 = _shape_param_2 = None
        permute_default: "f32[144, 288]" = torch.ops.aten.permute.default(primals_113, [1, 0]);  primals_113 = None
        permute_default_1: "f32[288, 144]" = torch.ops.aten.permute.default(permute_default, [1, 0]);  permute_default = None
        return (reshape_default_2, permute_default_1)


def _default_make_inputs():
    return [
    torch.randn([131072, 288], dtype=torch.float32, device='cuda'),
    torch.randn([131072, 288], dtype=torch.float32, device='cuda'),
    torch.randn([288, 144], dtype=torch.float32, device='cuda'),
    [512, 256, 288],  # _shape_param_0
    [512, 256, 288],  # _shape_param_1
    [131072, 288],  # _shape_param_2
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
