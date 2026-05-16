"""
Standalone repro captured via capture_hook.
Label: timm_deit_tiny_patch16_224.fb_in1k_training
Pattern hash: 549fbd1b11d8
Shape hash: f9dc69a7
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, getitem_132: "f32[32, 197, 1]", add_84: "f32[32, 197, 192]", getitem_133: "f32[32, 197, 1]", primals_150: "f32[192]", primals_151: "f32[192]", primals_152: "f32[1000, 192]", rsqrt_23: "f32[32, 197, 1]", rsqrt_22: "f32[32, 197, 1]", rsqrt_21: "f32[32, 197, 1]", rsqrt_20: "f32[32, 197, 1]", rsqrt_19: "f32[32, 197, 1]", rsqrt_18: "f32[32, 197, 1]", rsqrt_17: "f32[32, 197, 1]", rsqrt_16: "f32[32, 197, 1]", rsqrt_15: "f32[32, 197, 1]", rsqrt_14: "f32[32, 197, 1]", rsqrt_13: "f32[32, 197, 1]", rsqrt_12: "f32[32, 197, 1]", rsqrt_11: "f32[32, 197, 1]", rsqrt_10: "f32[32, 197, 1]", rsqrt_9: "f32[32, 197, 1]", rsqrt_8: "f32[32, 197, 1]", rsqrt_7: "f32[32, 197, 1]", rsqrt_6: "f32[32, 197, 1]", rsqrt_5: "f32[32, 197, 1]", rsqrt_4: "f32[32, 197, 1]", rsqrt_3: "f32[32, 197, 1]", rsqrt_2: "f32[32, 197, 1]", rsqrt_1: "f32[32, 197, 1]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        add_tensor: "f32[32, 197, 1]" = torch.ops.aten.add.Tensor(getitem_132, 1e-06);  getitem_132 = None
        rsqrt_default: "f32[32, 197, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        sub_tensor: "f32[32, 197, 192]" = torch.ops.aten.sub.Tensor(add_84, getitem_133);  add_84 = getitem_133 = None
        mul_tensor: "f32[32, 197, 192]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = None
        mul_tensor_1: "f32[32, 197, 192]" = torch.ops.aten.mul.Tensor(mul_tensor, primals_150);  mul_tensor = primals_150 = None
        add_tensor_1: "f32[32, 197, 192]" = torch.ops.aten.add.Tensor(mul_tensor_1, primals_151);  mul_tensor_1 = primals_151 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:696 in global_pool_nlc, code: x = x[:, 0]  # class token
        select_int: "f32[32, 192]" = torch.ops.aten.select.int(add_tensor_1, 1, 0);  add_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:1291 in forward_head, code: x = self.head_drop(x)
        clone_default: "f32[32, 192]" = torch.ops.aten.clone.default(select_int);  select_int = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:1292 in forward_head, code: return x if pre_logits else self.head(x)
        permute_default: "f32[192, 1000]" = torch.ops.aten.permute.default(primals_152, [1, 0]);  primals_152 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        div_tensor: "f32[32, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_default, 192);  rsqrt_default = None
        div_tensor_1: "f32[32, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_23, 192);  rsqrt_23 = None
        div_tensor_2: "f32[32, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_22, 192);  rsqrt_22 = None
        div_tensor_3: "f32[32, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_21, 192);  rsqrt_21 = None
        div_tensor_4: "f32[32, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_20, 192);  rsqrt_20 = None
        div_tensor_5: "f32[32, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_19, 192);  rsqrt_19 = None
        div_tensor_6: "f32[32, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_18, 192);  rsqrt_18 = None
        div_tensor_7: "f32[32, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_17, 192);  rsqrt_17 = None
        div_tensor_8: "f32[32, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_16, 192);  rsqrt_16 = None
        div_tensor_9: "f32[32, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_15, 192);  rsqrt_15 = None
        div_tensor_10: "f32[32, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_14, 192);  rsqrt_14 = None
        div_tensor_11: "f32[32, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_13, 192);  rsqrt_13 = None
        div_tensor_12: "f32[32, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_12, 192);  rsqrt_12 = None
        div_tensor_13: "f32[32, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_11, 192);  rsqrt_11 = None
        div_tensor_14: "f32[32, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_10, 192);  rsqrt_10 = None
        div_tensor_15: "f32[32, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_9, 192);  rsqrt_9 = None
        div_tensor_16: "f32[32, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_8, 192);  rsqrt_8 = None
        div_tensor_17: "f32[32, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_7, 192);  rsqrt_7 = None
        div_tensor_18: "f32[32, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_6, 192);  rsqrt_6 = None
        div_tensor_19: "f32[32, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_5, 192);  rsqrt_5 = None
        div_tensor_20: "f32[32, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_4, 192);  rsqrt_4 = None
        div_tensor_21: "f32[32, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_3, 192);  rsqrt_3 = None
        div_tensor_22: "f32[32, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_2, 192);  rsqrt_2 = None
        div_tensor_23: "f32[32, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_1, 192);  rsqrt_1 = None
        return (clone_default, permute_default, div_tensor, div_tensor_1, div_tensor_2, div_tensor_3, div_tensor_4, div_tensor_5, div_tensor_6, div_tensor_7, div_tensor_8, div_tensor_9, div_tensor_10, div_tensor_11, div_tensor_12, div_tensor_13, div_tensor_14, div_tensor_15, div_tensor_16, div_tensor_17, div_tensor_18, div_tensor_19, div_tensor_20, div_tensor_21, div_tensor_22, div_tensor_23)


def _default_make_inputs():
    return [
    torch.randn([32, 197, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32, 197, 192], dtype=torch.float32, device='cuda'),
    torch.randn([32, 197, 1], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([1000, 192], dtype=torch.float32, device='cuda'),
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
    torch.randn([32, 197, 1], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
