"""
Standalone repro captured via capture_hook.
Label: timm_convnextv2_nano.fcmae_ft_in22k_in1k_train
Pattern hash: be3cca3d51c6
Shape hash: 7a39551d
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([128, 2560, 7, 7], f32, stride=(125440, 1, 17920, 2560)), T([2560], f32), T([2560], f32))"

class Repro(torch.nn.Module):
    def forward(self, convolution_44: "f32[128, 2560, 7, 7]", primals_154: "f32[2560]", primals_155: "f32[2560]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:145 in forward, code: return F.gelu(input)
        mul_tensor: "f32[128, 2560, 7, 7]" = torch.ops.aten.mul.Tensor(convolution_44, 0.5)
        mul_tensor_1: "f32[128, 2560, 7, 7]" = torch.ops.aten.mul.Tensor(convolution_44, 0.7071067811865476);  convolution_44 = None
        erf_default: "f32[128, 2560, 7, 7]" = torch.ops.aten.erf.default(mul_tensor_1);  mul_tensor_1 = None
        add_tensor: "f32[128, 2560, 7, 7]" = torch.ops.aten.add.Tensor(erf_default, 1);  erf_default = None
        mul_tensor_2: "f32[128, 2560, 7, 7]" = torch.ops.aten.mul.Tensor(mul_tensor, add_tensor);  mul_tensor = add_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:45 in forward, code: x_g = x.norm(p=2, dim=self.spatial_dim, keepdim=True)
        pow_tensor_scalar: "f32[128, 2560, 7, 7]" = torch.ops.aten.pow.Tensor_Scalar(mul_tensor_2, 2)
        sum_dim_int_list: "f32[128, 2560, 1, 1]" = torch.ops.aten.sum.dim_IntList(pow_tensor_scalar, [2, 3], True);  pow_tensor_scalar = None
        pow_tensor_scalar_1: "f32[128, 2560, 1, 1]" = torch.ops.aten.pow.Tensor_Scalar(sum_dim_int_list, 0.5);  sum_dim_int_list = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:46 in forward, code: x_n = x_g / (x_g.mean(dim=self.channel_dim, keepdim=True) + self.eps)
        mean_dim: "f32[128, 1, 1, 1]" = torch.ops.aten.mean.dim(pow_tensor_scalar_1, [1], True)
        add_tensor_1: "f32[128, 1, 1, 1]" = torch.ops.aten.add.Tensor(mean_dim, 1e-06);  mean_dim = None
        div_tensor: "f32[128, 2560, 1, 1]" = torch.ops.aten.div.Tensor(pow_tensor_scalar_1, add_tensor_1);  pow_tensor_scalar_1 = add_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:47 in forward, code: return x + torch.addcmul(self.bias.view(self.wb_shape), self.weight.view(self.wb_shape), x * x_n)
        reshape_default: "f32[1, 2560, 1, 1]" = torch.ops.aten.reshape.default(primals_154, [1, -1, 1, 1]);  primals_154 = None
        reshape_default_1: "f32[1, 2560, 1, 1]" = torch.ops.aten.reshape.default(primals_155, [1, -1, 1, 1]);  primals_155 = None
        mul_tensor_3: "f32[128, 2560, 7, 7]" = torch.ops.aten.mul.Tensor(mul_tensor_2, div_tensor);  div_tensor = None
        addcmul_default: "f32[128, 2560, 7, 7]" = torch.ops.aten.addcmul.default(reshape_default, reshape_default_1, mul_tensor_3);  reshape_default = reshape_default_1 = mul_tensor_3 = None
        add_tensor_2: "f32[128, 2560, 7, 7]" = torch.ops.aten.add.Tensor(mul_tensor_2, addcmul_default);  mul_tensor_2 = addcmul_default = None
        return add_tensor_2


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
