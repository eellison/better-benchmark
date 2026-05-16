"""
Standalone repro captured via capture_hook.
Label: timm_visformer_small_training
Pattern hash: c77b8042f529
Shape hash: 2e33b51f
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, mm: "f32[32, 768]", _shape_param_0, _shape_param_1, sub_36: "f32[32, 768, 7, 7]", squeeze_82: "f32[768]", primals_203: "f32[768]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/adaptive_avgmax_pool.py:173 in forward, code: x = self.flatten(x)
        reshape_default: "f32[32, 768, 1, 1]" = torch.ops.aten.reshape.default(mm, _shape_param_0);  mm = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/adaptive_avgmax_pool.py:172 in forward, code: x = self.pool(x)
        expand_default: "f32[32, 768, 7, 7]" = torch.ops.aten.expand.default(reshape_default, _shape_param_1);  reshape_default = _shape_param_1 = None
        div_scalar: "f32[32, 768, 7, 7]" = torch.ops.aten.div.Scalar(expand_default, 49);  expand_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:468 in forward_features, code: x = self.norm(x)
        sum_dim_int_list: "f32[768]" = torch.ops.aten.sum.dim_IntList(div_scalar, [0, 2, 3])
        mul_tensor: "f32[32, 768, 7, 7]" = torch.ops.aten.mul.Tensor(div_scalar, sub_36)
        sum_dim_int_list_1: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [0, 2, 3]);  mul_tensor = None
        mul_tensor_1: "f32[768]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, 0.0006377551020408163);  sum_dim_int_list = None
        unsqueeze_default: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_tensor_1, 0);  mul_tensor_1 = None
        unsqueeze_default_1: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 2);  unsqueeze_default = None
        unsqueeze_default_2: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_1, 3);  unsqueeze_default_1 = None
        mul_tensor_2: "f32[768]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, 0.0006377551020408163);  sum_dim_int_list_1 = None
        mul_tensor_3: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_82, squeeze_82)
        mul_tensor_4: "f32[768]" = torch.ops.aten.mul.Tensor(mul_tensor_2, mul_tensor_3);  mul_tensor_2 = mul_tensor_3 = None
        unsqueeze_default_3: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_tensor_4, 0);  mul_tensor_4 = None
        unsqueeze_default_4: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_3, 2);  unsqueeze_default_3 = None
        unsqueeze_default_5: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 3);  unsqueeze_default_4 = None
        mul_tensor_5: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_82, primals_203);  squeeze_82 = primals_203 = None
        unsqueeze_default_6: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_tensor_5, 0);  mul_tensor_5 = None
        unsqueeze_default_7: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, 2);  unsqueeze_default_6 = None
        unsqueeze_default_8: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_7, 3);  unsqueeze_default_7 = None
        mul_tensor_6: "f32[32, 768, 7, 7]" = torch.ops.aten.mul.Tensor(sub_36, unsqueeze_default_5);  sub_36 = unsqueeze_default_5 = None
        sub_tensor: "f32[32, 768, 7, 7]" = torch.ops.aten.sub.Tensor(div_scalar, mul_tensor_6);  div_scalar = mul_tensor_6 = None
        sub_tensor_1: "f32[32, 768, 7, 7]" = torch.ops.aten.sub.Tensor(sub_tensor, unsqueeze_default_2);  sub_tensor = unsqueeze_default_2 = None
        mul_tensor_7: "f32[32, 768, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor_1, unsqueeze_default_8);  sub_tensor_1 = unsqueeze_default_8 = None
        return mul_tensor_7


def _default_make_inputs():
    return [
    torch.randn([32, 768], dtype=torch.float32, device='cuda'),
    [32, 768, 1, 1],  # _shape_param_0
    [32, 768, 7, 7],  # _shape_param_1
    torch.randn([32, 768, 7, 7], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
