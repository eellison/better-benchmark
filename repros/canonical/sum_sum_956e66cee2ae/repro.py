"""
Standalone repro captured via capture_hook.
Label: timm_convnextv2_nano.fcmae_ft_in22k_in1k_train
Pattern hash: 956e66cee2ae
Shape hash: 66d43a71
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([128, 640], f32), T([640], f32), T([128, 1, 1, 640], f32), T([128, 1, 1, 1], f32), S([128, 640, 1, 1]), S([128, 640, 7, 7]))"

class Repro(torch.nn.Module):
    def forward(self, mm: "f32[128, 640]", primals_158: "f32[640]", mul_92: "f32[128, 1, 1, 640]", div_14: "f32[128, 1, 1, 1]", _shape_param_0, _shape_param_1):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/classifier.py:214 in forward, code: x = self.flatten(x)
        reshape_default: "f32[128, 640, 1, 1]" = torch.ops.aten.reshape.default(mm, _shape_param_0);  mm = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:133 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_default: "f32[128, 1, 1, 640]" = torch.ops.aten.permute.default(reshape_default, [0, 2, 3, 1]);  reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_tensor: "f32[128, 1, 1, 640]" = torch.ops.aten.mul.Tensor(permute_default, primals_158);  permute_default = primals_158 = None
        mul_tensor_1: "f32[128, 1, 1, 640]" = torch.ops.aten.mul.Tensor(mul_tensor, 640)
        sum_dim_int_list: "f32[128, 1, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [3], True)
        mul_tensor_2: "f32[128, 1, 1, 640]" = torch.ops.aten.mul.Tensor(mul_tensor, mul_92);  mul_tensor = None
        sum_dim_int_list_1: "f32[128, 1, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [3], True);  mul_tensor_2 = None
        mul_tensor_3: "f32[128, 1, 1, 640]" = torch.ops.aten.mul.Tensor(mul_92, sum_dim_int_list_1);  mul_92 = sum_dim_int_list_1 = None
        sub_tensor: "f32[128, 1, 1, 640]" = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_1: "f32[128, 1, 1, 640]" = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor_3);  sub_tensor = mul_tensor_3 = None
        mul_tensor_4: "f32[128, 1, 1, 640]" = torch.ops.aten.mul.Tensor(div_14, sub_tensor_1);  div_14 = sub_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:128 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_default_1: "f32[128, 640, 1, 1]" = torch.ops.aten.permute.default(mul_tensor_4, [0, 3, 1, 2]);  mul_tensor_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/adaptive_avgmax_pool.py:172 in forward, code: x = self.pool(x)
        squeeze_dim: "f32[128, 640, 1]" = torch.ops.aten.squeeze.dim(permute_default_1, 3);  permute_default_1 = None
        squeeze_dim_1: "f32[128, 640]" = torch.ops.aten.squeeze.dim(squeeze_dim, 2);  squeeze_dim = None
        full_default: "f32[81920]" = torch.ops.aten.full.default([81920], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        as_strided_scatter_default: "f32[81920]" = torch.ops.aten.as_strided_scatter.default(full_default, squeeze_dim_1, [128, 640], [640, 1], 0);  full_default = squeeze_dim_1 = None
        as_strided_default: "f32[128, 640, 1, 1]" = torch.ops.aten.as_strided.default(as_strided_scatter_default, [128, 640, 1, 1], [640, 1, 1, 1], 0);  as_strided_scatter_default = None
        expand_default: "f32[128, 640, 7, 7]" = torch.ops.aten.expand.default(as_strided_default, _shape_param_1);  as_strided_default = _shape_param_1 = None
        div_scalar: "f32[128, 640, 7, 7]" = torch.ops.aten.div.Scalar(expand_default, 49);  expand_default = None
        return div_scalar


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
