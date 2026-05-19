"""
Standalone repro captured via capture_hook.
Label: timm_beit_base_patch16_224_train
Pattern hash: 7393bdc1b86d
Shape hash: 7ef0a5e0
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
    def forward(self, mm: "f32[128, 768]", primals_221: "f32[768]", mul_108: "f32[128, 768]", div: "f32[128, 1]", primals_214: "f32[768]", primals_219: "f32[768, 3072]", _shape_param_0, _shape_param_1):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_tensor: "f32[128, 768]" = torch.ops.aten.mul.Tensor(mm, primals_221);  mm = primals_221 = None
        mul_tensor_1: "f32[128, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, 768)
        sum_dim_int_list: "f32[128, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [1], True)
        mul_tensor_2: "f32[128, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, mul_108);  mul_tensor = None
        sum_dim_int_list_1: "f32[128, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [1], True);  mul_tensor_2 = None
        mul_tensor_3: "f32[128, 768]" = torch.ops.aten.mul.Tensor(mul_108, sum_dim_int_list_1);  mul_108 = sum_dim_int_list_1 = None
        sub_tensor: "f32[128, 768]" = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_1: "f32[128, 768]" = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor_3);  sub_tensor = mul_tensor_3 = None
        mul_tensor_4: "f32[128, 768]" = torch.ops.aten.mul.Tensor(div, sub_tensor_1);  div = sub_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:812 in forward_head, code: x = x[:, self.num_prefix_tokens:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
        unsqueeze_default: "f32[128, 1, 768]" = torch.ops.aten.unsqueeze.default(mul_tensor_4, 1);  mul_tensor_4 = None
        expand_default: "f32[128, 196, 768]" = torch.ops.aten.expand.default(unsqueeze_default, _shape_param_0);  unsqueeze_default = _shape_param_0 = None
        div_scalar: "f32[128, 196, 768]" = torch.ops.aten.div.Scalar(expand_default, 196);  expand_default = None
        full_default: "f32[128, 197, 768]" = torch.ops.aten.full.default([128, 197, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_scatter_default: "f32[128, 197, 768]" = torch.ops.aten.slice_scatter.default(full_default, div_scalar, 1, 1, 9223372036854775807);  full_default = div_scalar = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:389 in forward, code: x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
        mul_tensor_5: "f32[128, 197, 768]" = torch.ops.aten.mul.Tensor(slice_scatter_default, primals_214);  slice_scatter_default = primals_214 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        reshape_default: "f32[25216, 768]" = torch.ops.aten.reshape.default(mul_tensor_5, _shape_param_1);  mul_tensor_5 = _shape_param_1 = None
        permute_default: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_219, [1, 0]);  primals_219 = None
        permute_default_1: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_default, [1, 0]);  permute_default = None
        return (reshape_default, permute_default_1)


def _default_make_inputs():
    return [
    torch.randn([128, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([128, 768], dtype=torch.float32, device='cuda'),
    torch.randn([128, 1], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    [128, 196, 768],  # _shape_param_0
    [25216, 768],  # _shape_param_1
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
