"""
Standalone repro captured via capture_hook.
Label: timm_dm_nfnet_f0_train
Pattern hash: 8e96ea244bf1
Shape hash: a17bdef9
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
_shapes_config = "(T([16, 3, 3, 3], f32, stride=(27, 1, 9, 3)), T([16, 3, 3, 3], f32, stride=(27, 1, 9, 3)), T([1, 16, 1], f32), T([16], f32), T([16, 1, 1, 1], f32), S([1, 16, 27]), S([1, 16, 27]), S([16, 1, 1, 1]), S([16, 3, 3, 3]))"

class Repro(torch.nn.Module):
    def forward(self, getitem_355: "f32[16, 3, 3, 3]", primals_2: "f32[16, 3, 3, 3]", unsqueeze_506: "f32[1, 16, 1]", squeeze_1: "f32[16]", primals_3: "f32[16, 1, 1, 1]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:231 in forward, code: ).reshape_as(self.weight)
        clone_default: "f32[16, 3, 3, 3]" = torch.ops.aten.clone.default(getitem_355, memory_format = torch.contiguous_format);  getitem_355 = None
        reshape_default: "f32[1, 16, 27]" = torch.ops.aten.reshape.default(clone_default, _shape_param_0);  clone_default = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sum_dim_int_list: "f32[16]" = torch.ops.aten.sum.dim_IntList(reshape_default, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        clone_default_1: "f32[16, 3, 3, 3]" = torch.ops.aten.clone.default(primals_2, memory_format = torch.contiguous_format);  primals_2 = None
        reshape_default_1: "f32[1, 16, 27]" = torch.ops.aten.reshape.default(clone_default_1, _shape_param_1);  clone_default_1 = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sub_tensor: "f32[1, 16, 27]" = torch.ops.aten.sub.Tensor(reshape_default_1, unsqueeze_506);  reshape_default_1 = unsqueeze_506 = None
        mul_tensor: "f32[1, 16, 27]" = torch.ops.aten.mul.Tensor(reshape_default, sub_tensor)
        sum_dim_int_list_1: "f32[16]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [0, 2]);  mul_tensor = None
        mul_tensor_1: "f32[16]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, 0.037037037037037035);  sum_dim_int_list = None
        unsqueeze_default: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_tensor_1, 0);  mul_tensor_1 = None
        unsqueeze_default_1: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 2);  unsqueeze_default = None
        mul_tensor_2: "f32[16]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, 0.037037037037037035)
        mul_tensor_3: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_1, squeeze_1)
        mul_tensor_4: "f32[16]" = torch.ops.aten.mul.Tensor(mul_tensor_2, mul_tensor_3);  mul_tensor_2 = mul_tensor_3 = None
        unsqueeze_default_2: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_tensor_4, 0);  mul_tensor_4 = None
        unsqueeze_default_3: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, 2);  unsqueeze_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_tensor_5: "f32[16, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_3, 0.19245008972987526);  primals_3 = None
        reshape_default_2: "f32[16]" = torch.ops.aten.reshape.default(mul_tensor_5, [-1]);  mul_tensor_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        mul_tensor_6: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_1, reshape_default_2);  reshape_default_2 = None
        unsqueeze_default_4: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_tensor_6, 0);  mul_tensor_6 = None
        unsqueeze_default_5: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 2);  unsqueeze_default_4 = None
        mul_tensor_7: "f32[1, 16, 27]" = torch.ops.aten.mul.Tensor(sub_tensor, unsqueeze_default_3);  sub_tensor = unsqueeze_default_3 = None
        sub_tensor_1: "f32[1, 16, 27]" = torch.ops.aten.sub.Tensor(reshape_default, mul_tensor_7);  reshape_default = mul_tensor_7 = None
        sub_tensor_2: "f32[1, 16, 27]" = torch.ops.aten.sub.Tensor(sub_tensor_1, unsqueeze_default_1);  sub_tensor_1 = unsqueeze_default_1 = None
        mul_tensor_8: "f32[1, 16, 27]" = torch.ops.aten.mul.Tensor(sub_tensor_2, unsqueeze_default_5);  sub_tensor_2 = unsqueeze_default_5 = None
        mul_tensor_9: "f32[16]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, squeeze_1);  sum_dim_int_list_1 = squeeze_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        reshape_default_3: "f32[16, 1, 1, 1]" = torch.ops.aten.reshape.default(mul_tensor_9, _shape_param_2);  mul_tensor_9 = _shape_param_2 = None
        mul_tensor_10: "f32[16, 1, 1, 1]" = torch.ops.aten.mul.Tensor(reshape_default_3, 0.19245008972987526);  reshape_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        reshape_default_4: "f32[16, 3, 3, 3]" = torch.ops.aten.reshape.default(mul_tensor_8, _shape_param_3);  mul_tensor_8 = _shape_param_3 = None
        return (mul_tensor_10, reshape_default_4)



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
