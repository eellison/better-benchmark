"""
Standalone repro captured via capture_hook.
Label: timm_beit_base_patch16_224_infer
Pattern hash: 2340136d3c60
Shape hash: de324934
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
    def forward(self, addmm_47: "f32[25216, 768]", arg213_1: "f32[768]", add_79: "f32[128, 197, 768]", arg220_1: "f32[768]", arg221_1: "f32[768]", arg222_1: "f32[1000, 768]", _shape_param_0):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        reshape_default: "f32[128, 197, 768]" = torch.ops.aten.reshape.default(addmm_47, _shape_param_0);  addmm_47 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:389 in forward, code: x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
        mul_tensor: "f32[128, 197, 768]" = torch.ops.aten.mul.Tensor(arg213_1, reshape_default);  arg213_1 = reshape_default = None
        add_tensor: "f32[128, 197, 768]" = torch.ops.aten.add.Tensor(add_79, mul_tensor);  add_79 = mul_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:812 in forward_head, code: x = x[:, self.num_prefix_tokens:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
        slice_tensor: "f32[128, 196, 768]" = torch.ops.aten.slice.Tensor(add_tensor, 1, 1, 9223372036854775807);  add_tensor = None
        mean_dim: "f32[128, 768]" = torch.ops.aten.mean.dim(slice_tensor, [1]);  slice_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_correction = torch.ops.aten.var_mean.correction(mean_dim, [1], correction = 0, keepdim = True)
        getitem: "f32[128, 1]" = var_mean_correction[0]
        getitem_1: "f32[128, 1]" = var_mean_correction[1];  var_mean_correction = None
        sub_tensor: "f32[128, 768]" = torch.ops.aten.sub.Tensor(mean_dim, getitem_1);  mean_dim = getitem_1 = None
        add_tensor_1: "f32[128, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-06);  getitem = None
        rsqrt_default: "f32[128, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        mul_tensor_1: "f32[128, 768]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_2: "f32[128, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_1, arg220_1);  mul_tensor_1 = arg220_1 = None
        add_tensor_2: "f32[128, 768]" = torch.ops.aten.add.Tensor(mul_tensor_2, arg221_1);  mul_tensor_2 = arg221_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:815 in forward_head, code: return x if pre_logits else self.head(x)
        permute_default: "f32[768, 1000]" = torch.ops.aten.permute.default(arg222_1, [1, 0]);  arg222_1 = None
        return (add_tensor_2, permute_default)


def _default_make_inputs():
    return [
    torch.randn([25216, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([128, 197, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([1000, 768], dtype=torch.float32, device='cuda'),
    [128, 197, 768],  # _shape_param_0
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
