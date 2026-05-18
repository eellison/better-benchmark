"""
Standalone repro captured via capture_hook.
Label: timm_beit_base_patch16_224_training
Pattern hash: cdae6aa7cad7
Shape hash: 4c240213
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
    def forward(self, getitem_132: "f32[32, 1]", mean: "f32[32, 768]", getitem_133: "f32[32, 1]", primals_221: "f32[768]", primals_222: "f32[768]", primals_223: "f32[1000, 768]", rsqrt_23: "f32[32, 197, 1]", rsqrt_22: "f32[32, 197, 1]", rsqrt_21: "f32[32, 197, 1]", rsqrt_20: "f32[32, 197, 1]", rsqrt_19: "f32[32, 197, 1]", rsqrt_18: "f32[32, 197, 1]", rsqrt_17: "f32[32, 197, 1]", rsqrt_16: "f32[32, 197, 1]", rsqrt_15: "f32[32, 197, 1]", rsqrt_14: "f32[32, 197, 1]", rsqrt_13: "f32[32, 197, 1]", rsqrt_12: "f32[32, 197, 1]", rsqrt_11: "f32[32, 197, 1]", rsqrt_10: "f32[32, 197, 1]", rsqrt_9: "f32[32, 197, 1]", rsqrt_8: "f32[32, 197, 1]", rsqrt_7: "f32[32, 197, 1]", rsqrt_6: "f32[32, 197, 1]", rsqrt_5: "f32[32, 197, 1]", rsqrt_4: "f32[32, 197, 1]", rsqrt_3: "f32[32, 197, 1]", rsqrt_2: "f32[32, 197, 1]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        add_tensor: "f32[32, 1]" = torch.ops.aten.add.Tensor(getitem_132, 1e-06);  getitem_132 = None
        rsqrt_default: "f32[32, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        sub_tensor: "f32[32, 768]" = torch.ops.aten.sub.Tensor(mean, getitem_133);  mean = getitem_133 = None
        mul_tensor: "f32[32, 768]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = None
        mul_tensor_1: "f32[32, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, primals_221);  mul_tensor = primals_221 = None
        add_tensor_1: "f32[32, 768]" = torch.ops.aten.add.Tensor(mul_tensor_1, primals_222);  mul_tensor_1 = primals_222 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:815 in forward_head, code: return x if pre_logits else self.head(x)
        permute_default: "f32[768, 1000]" = torch.ops.aten.permute.default(primals_223, [1, 0]);  primals_223 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        div_tensor: "f32[32, 1]" = torch.ops.aten.div.Tensor(rsqrt_default, 768);  rsqrt_default = None
        div_tensor_1: "f32[32, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_23, 768);  rsqrt_23 = None
        div_tensor_2: "f32[32, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_22, 768);  rsqrt_22 = None
        div_tensor_3: "f32[32, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_21, 768);  rsqrt_21 = None
        div_tensor_4: "f32[32, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_20, 768);  rsqrt_20 = None
        div_tensor_5: "f32[32, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_19, 768);  rsqrt_19 = None
        div_tensor_6: "f32[32, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_18, 768);  rsqrt_18 = None
        div_tensor_7: "f32[32, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_17, 768);  rsqrt_17 = None
        div_tensor_8: "f32[32, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_16, 768);  rsqrt_16 = None
        div_tensor_9: "f32[32, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_15, 768);  rsqrt_15 = None
        div_tensor_10: "f32[32, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_14, 768);  rsqrt_14 = None
        div_tensor_11: "f32[32, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_13, 768);  rsqrt_13 = None
        div_tensor_12: "f32[32, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_12, 768);  rsqrt_12 = None
        div_tensor_13: "f32[32, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_11, 768);  rsqrt_11 = None
        div_tensor_14: "f32[32, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_10, 768);  rsqrt_10 = None
        div_tensor_15: "f32[32, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_9, 768);  rsqrt_9 = None
        div_tensor_16: "f32[32, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_8, 768);  rsqrt_8 = None
        div_tensor_17: "f32[32, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_7, 768);  rsqrt_7 = None
        div_tensor_18: "f32[32, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_6, 768);  rsqrt_6 = None
        div_tensor_19: "f32[32, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_5, 768);  rsqrt_5 = None
        div_tensor_20: "f32[32, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_4, 768);  rsqrt_4 = None
        div_tensor_21: "f32[32, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_3, 768);  rsqrt_3 = None
        div_tensor_22: "f32[32, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_2, 768);  rsqrt_2 = None
        return (add_tensor_1, permute_default, div_tensor, div_tensor_1, div_tensor_2, div_tensor_3, div_tensor_4, div_tensor_5, div_tensor_6, div_tensor_7, div_tensor_8, div_tensor_9, div_tensor_10, div_tensor_11, div_tensor_12, div_tensor_13, div_tensor_14, div_tensor_15, div_tensor_16, div_tensor_17, div_tensor_18, div_tensor_19, div_tensor_20, div_tensor_21, div_tensor_22)


def _default_make_inputs():
    return [
    torch.randn([32, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32, 768], dtype=torch.float32, device='cuda'),
    torch.randn([32, 1], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([1000, 768], dtype=torch.float32, device='cuda'),
    torch.randn([32, 197, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32, 197, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32, 197, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32, 197, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32, 197, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32, 197, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32, 197, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32, 197, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32, 197, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32, 197, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32, 197, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32, 197, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32, 197, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32, 197, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32, 197, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32, 197, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32, 197, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32, 197, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32, 197, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32, 197, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32, 197, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32, 197, 1], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
