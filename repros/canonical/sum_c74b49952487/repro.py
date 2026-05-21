"""
Standalone repro captured via capture_hook.
Label: timm_mobilenetv3_large_100_train
Pattern hash: c74b49952487
Shape hash: 25f6ca38
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([512, 480, 14, 14], f32, stride=(94080, 1, 6720, 480)), T([1, 480, 1, 1], f32), T([1, 480, 1, 1], f32), T([480], f32), T([480], f32), T([512, 480, 14, 14], f32, stride=(94080, 1, 6720, 480)), T([512, 480, 1, 1], f32), T([], f32))"

class Repro(torch.nn.Module):
    def forward(self, convolution_37: "f32[512, 480, 14, 14]", getitem_63: "f32[1, 480, 1, 1]", rsqrt_31: "f32[1, 480, 1, 1]", primals_204: "f32[480]", primals_205: "f32[480]", getitem_158: "f32[512, 480, 14, 14]", convolution_39: "f32[512, 480, 1, 1]", full_default: "f32[]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_tensor: "f32[512, 480, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_37, getitem_63);  convolution_37 = getitem_63 = None
        mul_tensor: "f32[512, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_31);  sub_tensor = rsqrt_31 = None
        unsqueeze_default: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_204, -1);  primals_204 = None
        unsqueeze_default_1: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_1: "f32[512, 480, 14, 14]" = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default_1);  mul_tensor = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_205, -1);  primals_205 = None
        unsqueeze_default_3: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor: "f32[512, 480, 14, 14]" = torch.ops.aten.add.Tensor(mul_tensor_1, unsqueeze_default_3);  mul_tensor_1 = unsqueeze_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        add_tensor_1: "f32[512, 480, 14, 14]" = torch.ops.aten.add.Tensor(add_tensor, 3)
        clamp_min_default: "f32[512, 480, 14, 14]" = torch.ops.aten.clamp_min.default(add_tensor_1, 0);  add_tensor_1 = None
        clamp_max_default: "f32[512, 480, 14, 14]" = torch.ops.aten.clamp_max.default(clamp_min_default, 6);  clamp_min_default = None
        mul_tensor_2: "f32[512, 480, 14, 14]" = torch.ops.aten.mul.Tensor(add_tensor, clamp_max_default);  add_tensor = clamp_max_default = None
        div_tensor: "f32[512, 480, 14, 14]" = torch.ops.aten.div.Tensor(mul_tensor_2, 6);  mul_tensor_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:83 in forward, code: return x * self.gate(x_se)
        mul_tensor_3: "f32[512, 480, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_158, div_tensor);  getitem_158 = div_tensor = None
        sum_dim_int_list: "f32[512, 480, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_3, [2, 3], True);  mul_tensor_3 = None
        gt_scalar: "b8[512, 480, 1, 1]" = torch.ops.aten.gt.Scalar(convolution_39, -3.0)
        lt_scalar: "b8[512, 480, 1, 1]" = torch.ops.aten.lt.Scalar(convolution_39, 3.0);  convolution_39 = None
        bitwise_and_tensor: "b8[512, 480, 1, 1]" = torch.ops.aten.bitwise_and.Tensor(gt_scalar, lt_scalar);  gt_scalar = lt_scalar = None
        mul_tensor_4: "f32[512, 480, 1, 1]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, 0.16666666666666666);  sum_dim_int_list = None
        where_self: "f32[512, 480, 1, 1]" = torch.ops.aten.where.self(bitwise_and_tensor, mul_tensor_4, full_default);  bitwise_and_tensor = mul_tensor_4 = full_default = None
        return where_self



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
