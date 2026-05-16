"""
Standalone repro captured via capture_hook.
Label: timm_adv_inception_v3_training
Pattern hash: b0237cfc6dbd
Shape hash: 32f3fbf2
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, getitem_126: "f32[1, 192, 1, 1]", convolution_60: "f32[32, 192, 17, 17]", getitem_127: "f32[1, 192, 1, 1]", primals_366: "f32[192]", primals_367: "f32[192]", getitem_132: "f32[1, 192, 1, 1]", convolution_63: "f32[32, 192, 17, 17]", getitem_133: "f32[1, 192, 1, 1]", primals_384: "f32[192]", primals_385: "f32[192]", getitem_142: "f32[1, 192, 1, 1]", convolution_68: "f32[32, 192, 17, 17]", getitem_143: "f32[1, 192, 1, 1]", primals_414: "f32[192]", primals_415: "f32[192]", getitem_144: "f32[1, 192, 1, 1]", convolution_69: "f32[32, 192, 17, 17]", getitem_145: "f32[1, 192, 1, 1]", primals_420: "f32[192]", primals_421: "f32[192]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        add_tensor: "f32[1, 192, 1, 1]" = torch.ops.aten.add.Tensor(getitem_126, 0.001);  getitem_126 = None
        rsqrt_default: "f32[1, 192, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        sub_tensor: "f32[32, 192, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_60, getitem_127);  convolution_60 = getitem_127 = None
        mul_tensor: "f32[32, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        unsqueeze_default: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_366, -1);  primals_366 = None
        unsqueeze_default_1: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_1: "f32[32, 192, 17, 17]" = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default_1);  mul_tensor = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_367, -1);  primals_367 = None
        unsqueeze_default_3: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor_1: "f32[32, 192, 17, 17]" = torch.ops.aten.add.Tensor(mul_tensor_1, unsqueeze_default_3);  mul_tensor_1 = unsqueeze_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_default: "f32[32, 192, 17, 17]" = torch.ops.aten.relu.default(add_tensor_1);  add_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        add_tensor_2: "f32[1, 192, 1, 1]" = torch.ops.aten.add.Tensor(getitem_132, 0.001);  getitem_132 = None
        rsqrt_default_1: "f32[1, 192, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor_2);  add_tensor_2 = None
        sub_tensor_1: "f32[32, 192, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_63, getitem_133);  convolution_63 = getitem_133 = None
        mul_tensor_2: "f32[32, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_tensor_1, rsqrt_default_1);  sub_tensor_1 = rsqrt_default_1 = None
        unsqueeze_default_4: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_384, -1);  primals_384 = None
        unsqueeze_default_5: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, -1);  unsqueeze_default_4 = None
        mul_tensor_3: "f32[32, 192, 17, 17]" = torch.ops.aten.mul.Tensor(mul_tensor_2, unsqueeze_default_5);  mul_tensor_2 = unsqueeze_default_5 = None
        unsqueeze_default_6: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_385, -1);  primals_385 = None
        unsqueeze_default_7: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, -1);  unsqueeze_default_6 = None
        add_tensor_3: "f32[32, 192, 17, 17]" = torch.ops.aten.add.Tensor(mul_tensor_3, unsqueeze_default_7);  mul_tensor_3 = unsqueeze_default_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_default_1: "f32[32, 192, 17, 17]" = torch.ops.aten.relu.default(add_tensor_3);  add_tensor_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        add_tensor_4: "f32[1, 192, 1, 1]" = torch.ops.aten.add.Tensor(getitem_142, 0.001);  getitem_142 = None
        rsqrt_default_2: "f32[1, 192, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor_4);  add_tensor_4 = None
        sub_tensor_2: "f32[32, 192, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_68, getitem_143);  convolution_68 = getitem_143 = None
        mul_tensor_4: "f32[32, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_tensor_2, rsqrt_default_2);  sub_tensor_2 = rsqrt_default_2 = None
        unsqueeze_default_8: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_414, -1);  primals_414 = None
        unsqueeze_default_9: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_8, -1);  unsqueeze_default_8 = None
        mul_tensor_5: "f32[32, 192, 17, 17]" = torch.ops.aten.mul.Tensor(mul_tensor_4, unsqueeze_default_9);  mul_tensor_4 = unsqueeze_default_9 = None
        unsqueeze_default_10: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_415, -1);  primals_415 = None
        unsqueeze_default_11: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_10, -1);  unsqueeze_default_10 = None
        add_tensor_5: "f32[32, 192, 17, 17]" = torch.ops.aten.add.Tensor(mul_tensor_5, unsqueeze_default_11);  mul_tensor_5 = unsqueeze_default_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_default_2: "f32[32, 192, 17, 17]" = torch.ops.aten.relu.default(add_tensor_5);  add_tensor_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        add_tensor_6: "f32[1, 192, 1, 1]" = torch.ops.aten.add.Tensor(getitem_144, 0.001);  getitem_144 = None
        rsqrt_default_3: "f32[1, 192, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor_6);  add_tensor_6 = None
        sub_tensor_3: "f32[32, 192, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_69, getitem_145);  convolution_69 = getitem_145 = None
        mul_tensor_6: "f32[32, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_tensor_3, rsqrt_default_3);  sub_tensor_3 = rsqrt_default_3 = None
        unsqueeze_default_12: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_420, -1);  primals_420 = None
        unsqueeze_default_13: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_12, -1);  unsqueeze_default_12 = None
        mul_tensor_7: "f32[32, 192, 17, 17]" = torch.ops.aten.mul.Tensor(mul_tensor_6, unsqueeze_default_13);  mul_tensor_6 = unsqueeze_default_13 = None
        unsqueeze_default_14: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_421, -1);  primals_421 = None
        unsqueeze_default_15: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_14, -1);  unsqueeze_default_14 = None
        add_tensor_7: "f32[32, 192, 17, 17]" = torch.ops.aten.add.Tensor(mul_tensor_7, unsqueeze_default_15);  mul_tensor_7 = unsqueeze_default_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_default_3: "f32[32, 192, 17, 17]" = torch.ops.aten.relu.default(add_tensor_7);  add_tensor_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/inception_v3.py:152 in forward, code: return torch.cat(outputs, 1)
        cat_default: "f32[32, 768, 17, 17]" = torch.ops.aten.cat.default([relu_default, relu_default_1, relu_default_2, relu_default_3], 1);  relu_default = relu_default_1 = relu_default_2 = relu_default_3 = None
        return cat_default


def _default_make_inputs():
    return [
    torch.randn([1, 192, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32, 192, 17, 17], dtype=torch.float32, device='cuda'),
    torch.randn([1, 192, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([1, 192, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32, 192, 17, 17], dtype=torch.float32, device='cuda'),
    torch.randn([1, 192, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([1, 192, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32, 192, 17, 17], dtype=torch.float32, device='cuda'),
    torch.randn([1, 192, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([1, 192, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32, 192, 17, 17], dtype=torch.float32, device='cuda'),
    torch.randn([1, 192, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
