"""
Standalone repro captured via capture_hook.
Label: timm_adv_inception_v3_train
Pattern hash: 17d9ba8b14ab
Shape hash: 5f4d27f8
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
_shapes_config = "(T([128, 384, 8, 8], f32, stride=(24576, 1, 3072, 384)), T([128, 384, 8, 8], f32, stride=(24576, 1, 3072, 384)), T([128, 384, 8, 8], f32, stride=(24576, 1, 3072, 384)), T([], f32), T([128, 384, 8, 8], f32, stride=(24576, 1, 3072, 384)), T([1, 384, 1, 1], f32), T([384], f32), T([384], f32))"

class Repro(torch.nn.Module):
    def forward(self, getitem_238: "f32[128, 384, 8, 8]", getitem_241: "f32[128, 384, 8, 8]", relu_77: "f32[128, 384, 8, 8]", full_default: "f32[]", convolution_77: "f32[128, 384, 8, 8]", unsqueeze_570: "f32[1, 384, 1, 1]", squeeze_232: "f32[384]", primals_468: "f32[384]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        add_tensor: "f32[128, 384, 8, 8]" = torch.ops.aten.add.Tensor(getitem_238, getitem_241);  getitem_238 = getitem_241 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        le_scalar: "b8[128, 384, 8, 8]" = torch.ops.aten.le.Scalar(relu_77, 0);  relu_77 = None
        where_self: "f32[128, 384, 8, 8]" = torch.ops.aten.where.self(le_scalar, full_default, add_tensor);  le_scalar = full_default = add_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sum_dim_int_list: "f32[384]" = torch.ops.aten.sum.dim_IntList(where_self, [0, 2, 3])
        sub_tensor: "f32[128, 384, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_77, unsqueeze_570);  convolution_77 = unsqueeze_570 = None
        mul_tensor: "f32[128, 384, 8, 8]" = torch.ops.aten.mul.Tensor(where_self, sub_tensor)
        sum_dim_int_list_1: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [0, 2, 3]);  mul_tensor = None
        mul_tensor_1: "f32[384]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, 0.0001220703125);  sum_dim_int_list = None
        unsqueeze_default: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_tensor_1, 0);  mul_tensor_1 = None
        unsqueeze_default_1: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 2);  unsqueeze_default = None
        unsqueeze_default_2: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_1, 3);  unsqueeze_default_1 = None
        mul_tensor_2: "f32[384]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, 0.0001220703125);  sum_dim_int_list_1 = None
        mul_tensor_3: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_232, squeeze_232)
        mul_tensor_4: "f32[384]" = torch.ops.aten.mul.Tensor(mul_tensor_2, mul_tensor_3);  mul_tensor_2 = mul_tensor_3 = None
        unsqueeze_default_3: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_tensor_4, 0);  mul_tensor_4 = None
        unsqueeze_default_4: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_3, 2);  unsqueeze_default_3 = None
        unsqueeze_default_5: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 3);  unsqueeze_default_4 = None
        mul_tensor_5: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_232, primals_468);  squeeze_232 = primals_468 = None
        unsqueeze_default_6: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_tensor_5, 0);  mul_tensor_5 = None
        unsqueeze_default_7: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, 2);  unsqueeze_default_6 = None
        unsqueeze_default_8: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_7, 3);  unsqueeze_default_7 = None
        mul_tensor_6: "f32[128, 384, 8, 8]" = torch.ops.aten.mul.Tensor(sub_tensor, unsqueeze_default_5);  sub_tensor = unsqueeze_default_5 = None
        sub_tensor_1: "f32[128, 384, 8, 8]" = torch.ops.aten.sub.Tensor(where_self, mul_tensor_6);  where_self = mul_tensor_6 = None
        sub_tensor_2: "f32[128, 384, 8, 8]" = torch.ops.aten.sub.Tensor(sub_tensor_1, unsqueeze_default_2);  sub_tensor_1 = unsqueeze_default_2 = None
        mul_tensor_7: "f32[128, 384, 8, 8]" = torch.ops.aten.mul.Tensor(sub_tensor_2, unsqueeze_default_8);  sub_tensor_2 = unsqueeze_default_8 = None
        return mul_tensor_7



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
