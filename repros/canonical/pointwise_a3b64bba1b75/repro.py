"""
Standalone repro captured via capture_hook.
Label: timm_convnextv2_nano.fcmae_ft_in22k_in1k_training
Pattern hash: a3b64bba1b75
Shape hash: 49ae23a1
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_prelude import *  # noqa: F401,F403
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, getitem_36: "f32[32, 1, 1, 1]", permute_36: "f32[32, 1, 1, 640]", getitem_37: "f32[32, 1, 1, 1]", primals_158: "f32[640]", primals_159: "f32[640]", _shape_param_0, primals_160: "f32[1000, 640]", rsqrt_15: "f32[32, 14, 14, 1]", rsqrt_6: "f32[32, 28, 28, 1]", rsqrt_3: "f32[32, 56, 56, 1]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        add_tensor: "f32[32, 1, 1, 1]" = torch.ops.aten.add.Tensor(getitem_36, 1e-06);  getitem_36 = None
        rsqrt_default: "f32[32, 1, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        sub_tensor: "f32[32, 1, 1, 640]" = torch.ops.aten.sub.Tensor(permute_36, getitem_37);  permute_36 = getitem_37 = None
        mul_tensor: "f32[32, 1, 1, 640]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = None
        mul_tensor_1: "f32[32, 1, 1, 640]" = torch.ops.aten.mul.Tensor(mul_tensor, primals_158);  mul_tensor = primals_158 = None
        add_tensor_1: "f32[32, 1, 1, 640]" = torch.ops.aten.add.Tensor(mul_tensor_1, primals_159);  mul_tensor_1 = primals_159 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:133 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_default: "f32[32, 640, 1, 1]" = torch.ops.aten.permute.default(add_tensor_1, [0, 3, 1, 2]);  add_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/classifier.py:214 in forward, code: x = self.flatten(x)
        reshape_default: "f32[32, 640]" = torch.ops.aten.reshape.default(permute_default, _shape_param_0);  permute_default = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/classifier.py:219 in forward, code: x = self.fc(x)
        permute_default_1: "f32[640, 1000]" = torch.ops.aten.permute.default(primals_160, [1, 0]);  primals_160 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        div_tensor: "f32[32, 1, 1, 1]" = torch.ops.aten.div.Tensor(rsqrt_default, 640);  rsqrt_default = None
        div_tensor_1: "f32[32, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_15, 320);  rsqrt_15 = None
        div_tensor_2: "f32[32, 28, 28, 1]" = torch.ops.aten.div.Tensor(rsqrt_6, 160);  rsqrt_6 = None
        div_tensor_3: "f32[32, 56, 56, 1]" = torch.ops.aten.div.Tensor(rsqrt_3, 80);  rsqrt_3 = None
        return (reshape_default, permute_default_1, div_tensor, div_tensor_1, div_tensor_2, div_tensor_3)


def _default_make_inputs():
    return [
    torch.randn([32, 1, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32, 1, 1, 640], dtype=torch.float32, device='cuda'),
    torch.randn([32, 1, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([640], dtype=torch.float32, device='cuda'),
    torch.randn([640], dtype=torch.float32, device='cuda'),
    [32, 640],  # _shape_param_0
    torch.randn([1000, 640], dtype=torch.float32, device='cuda'),
    torch.randn([32, 14, 14, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32, 28, 28, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32, 56, 56, 1], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
