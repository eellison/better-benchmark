"""
Standalone repro captured via capture_hook.
Label: torchbench_timm_resnest_train
Pattern hash: f7795e0ab878
Shape hash: c9d1356e
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
_shapes_config = "(T([32, 128, 28, 28], f32), T([32, 128, 56, 56], f32), T([32, 256, 56, 56], f32), T([1, 256, 1, 1], f32), T([1, 256, 1, 1], f32), T([256], f32), T([256], f32), T([32, 256, 1, 1], f32), S([32, 2, 128, 56, 56]), S([32, 2, 128, 56, 56]), S([32, 1, 2, -1]), S([32, 256, 1, 1]), S([32, 256]), S([32, 2, 1, 128]), S([32, 256, 1, 1]))"

class Repro(torch.nn.Module):
    def forward(self, getitem_87: "f32[32, 128, 28, 28]", sum_6: "f32[32, 128, 56, 56]", convolution_10: "f32[32, 256, 56, 56]", getitem_21: "f32[1, 256, 1, 1]", rsqrt_9: "f32[1, 256, 1, 1]", primals_63: "f32[256]", primals_64: "f32[256]", convolution_12: "f32[32, 256, 1, 1]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/resnest.py:130 in forward, code: out = self.avd_last(out)
        avg_pool2d_backward_default: "f32[32, 128, 56, 56]" = torch.ops.aten.avg_pool2d_backward.default(getitem_87, sum_6, [3, 3], [2, 2], [1, 1], False, True, None);  getitem_87 = sum_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:109 in forward, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
        unsqueeze_default: "f32[32, 1, 128, 56, 56]" = torch.ops.aten.unsqueeze.default(avg_pool2d_backward_default, 1);  avg_pool2d_backward_default = None
        expand_default: "f32[32, 2, 128, 56, 56]" = torch.ops.aten.expand.default(unsqueeze_default, _shape_param_0);  unsqueeze_default = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:91 in forward, code: x = self.bn0(x)
        sub_tensor: "f32[32, 256, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_10, getitem_21);  convolution_10 = getitem_21 = None
        mul_tensor: "f32[32, 256, 56, 56]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_9);  sub_tensor = rsqrt_9 = None
        unsqueeze_default_1: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_63, -1);  primals_63 = None
        unsqueeze_default_2: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_1, -1);  unsqueeze_default_1 = None
        mul_tensor_1: "f32[32, 256, 56, 56]" = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default_2);  mul_tensor = unsqueeze_default_2 = None
        unsqueeze_default_3: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_64, -1);  primals_64 = None
        unsqueeze_default_4: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_3, -1);  unsqueeze_default_3 = None
        add_tensor: "f32[32, 256, 56, 56]" = torch.ops.aten.add.Tensor(mul_tensor_1, unsqueeze_default_4);  mul_tensor_1 = unsqueeze_default_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:93 in forward, code: x = self.act0(x)
        relu_default: "f32[32, 256, 56, 56]" = torch.ops.aten.relu.default(add_tensor);  add_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:97 in forward, code: x = x.reshape((B, self.radix, RC // self.radix, H, W))
        reshape_default: "f32[32, 2, 128, 56, 56]" = torch.ops.aten.reshape.default(relu_default, _shape_param_1);  relu_default = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:109 in forward, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
        mul_tensor_2: "f32[32, 2, 128, 56, 56]" = torch.ops.aten.mul.Tensor(expand_default, reshape_default);  expand_default = reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:27 in forward, code: x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
        reshape_default_1: "f32[32, 1, 2, 128]" = torch.ops.aten.reshape.default(convolution_12, _shape_param_2);  convolution_12 = _shape_param_2 = None
        permute_default: "f32[32, 2, 1, 128]" = torch.ops.aten.permute.default(reshape_default_1, [0, 2, 1, 3]);  reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:28 in forward, code: x = F.softmax(x, dim=1)
        amax_default: "f32[32, 1, 1, 128]" = torch.ops.aten.amax.default(permute_default, [1], True)
        sub_tensor_1: "f32[32, 2, 1, 128]" = torch.ops.aten.sub.Tensor(permute_default, amax_default);  permute_default = amax_default = None
        exp_default: "f32[32, 2, 1, 128]" = torch.ops.aten.exp.default(sub_tensor_1);  sub_tensor_1 = None
        sum_dim_int_list: "f32[32, 1, 1, 128]" = torch.ops.aten.sum.dim_IntList(exp_default, [1], True)
        div_tensor: "f32[32, 2, 1, 128]" = torch.ops.aten.div.Tensor(exp_default, sum_dim_int_list);  exp_default = sum_dim_int_list = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:109 in forward, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
        sum_dim_int_list_1: "f32[32, 2, 128, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [3, 4], True);  mul_tensor_2 = None
        reshape_default_2: "f32[32, 256, 1, 1]" = torch.ops.aten.reshape.default(sum_dim_int_list_1, _shape_param_3);  sum_dim_int_list_1 = _shape_param_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:107 in forward, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
        reshape_default_3: "f32[32, 256]" = torch.ops.aten.reshape.default(reshape_default_2, _shape_param_4);  reshape_default_2 = _shape_param_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:29 in forward, code: x = x.reshape(batch, -1)
        reshape_default_4: "f32[32, 2, 1, 128]" = torch.ops.aten.reshape.default(reshape_default_3, _shape_param_5);  reshape_default_3 = _shape_param_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:28 in forward, code: x = F.softmax(x, dim=1)
        mul_tensor_3: "f32[32, 2, 1, 128]" = torch.ops.aten.mul.Tensor(reshape_default_4, div_tensor);  reshape_default_4 = None
        sum_dim_int_list_2: "f32[32, 1, 1, 128]" = torch.ops.aten.sum.dim_IntList(mul_tensor_3, [1], True)
        neg_default: "f32[32, 2, 1, 128]" = torch.ops.aten.neg.default(div_tensor);  div_tensor = None
        fma_default: "f32[32, 2, 1, 128]" = torch.ops.prims.fma.default(neg_default, sum_dim_int_list_2, mul_tensor_3);  neg_default = sum_dim_int_list_2 = mul_tensor_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:27 in forward, code: x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
        permute_default_1: "f32[32, 1, 2, 128]" = torch.ops.aten.permute.default(fma_default, [0, 2, 1, 3]);  fma_default = None
        reshape_default_5: "f32[32, 256, 1, 1]" = torch.ops.aten.reshape.default(permute_default_1, _shape_param_6);  permute_default_1 = _shape_param_6 = None
        return reshape_default_5



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
