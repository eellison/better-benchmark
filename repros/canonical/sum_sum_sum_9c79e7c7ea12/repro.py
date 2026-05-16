"""
Standalone repro captured via capture_hook.
Label: timm_visformer_small_training
Pattern hash: 9c79e7c7ea12
Shape hash: 51f10fbc
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, getitem_242: "f32[32, 192, 28, 28]", convolution_1: "f32[32, 192, 28, 28]", getitem_3: "f32[1, 192, 1, 1]", rsqrt_1: "f32[1, 192, 1, 1]", primals_13: "f32[192]", primals_14: "f32[192]", primals_15: "f32[1, 192, 28, 28]", unsqueeze_414: "f32[1, 192, 1, 1]", squeeze_7: "f32[192]", primals_19: "f32[192]", add_249: "f32[32, 192, 28, 28]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:177 in forward, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
        sum_dim_int_list: "f32[192]" = torch.ops.aten.sum.dim_IntList(getitem_242, [0, 2, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/patch_embed.py:141 in forward, code: x = self.norm(x)
        sub_tensor: "f32[32, 192, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_1, getitem_3)
        mul_tensor: "f32[32, 192, 28, 28]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_1);  sub_tensor = None
        unsqueeze_default: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_13, -1)
        unsqueeze_default_1: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_1: "f32[32, 192, 28, 28]" = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default_1);  mul_tensor = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_14, -1);  primals_14 = None
        unsqueeze_default_3: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor: "f32[32, 192, 28, 28]" = torch.ops.aten.add.Tensor(mul_tensor_1, unsqueeze_default_3);  mul_tensor_1 = unsqueeze_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:442 in forward_features, code: x = self.pos_drop(x + self.pos_embed1)
        add_tensor_1: "f32[32, 192, 28, 28]" = torch.ops.aten.add.Tensor(add_tensor, primals_15);  add_tensor = primals_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:177 in forward, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
        sub_tensor_1: "f32[32, 192, 28, 28]" = torch.ops.aten.sub.Tensor(add_tensor_1, unsqueeze_414);  add_tensor_1 = unsqueeze_414 = None
        mul_tensor_2: "f32[32, 192, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_242, sub_tensor_1)
        sum_dim_int_list_1: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [0, 2, 3]);  mul_tensor_2 = None
        mul_tensor_3: "f32[192]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, 3.985969387755102e-05);  sum_dim_int_list = None
        unsqueeze_default_4: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_tensor_3, 0);  mul_tensor_3 = None
        unsqueeze_default_5: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 2);  unsqueeze_default_4 = None
        unsqueeze_default_6: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_5, 3);  unsqueeze_default_5 = None
        mul_tensor_4: "f32[192]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, 3.985969387755102e-05);  sum_dim_int_list_1 = None
        mul_tensor_5: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_7, squeeze_7)
        mul_tensor_6: "f32[192]" = torch.ops.aten.mul.Tensor(mul_tensor_4, mul_tensor_5);  mul_tensor_4 = mul_tensor_5 = None
        unsqueeze_default_7: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_tensor_6, 0);  mul_tensor_6 = None
        unsqueeze_default_8: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_7, 2);  unsqueeze_default_7 = None
        unsqueeze_default_9: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_8, 3);  unsqueeze_default_8 = None
        mul_tensor_7: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_7, primals_19);  squeeze_7 = primals_19 = None
        unsqueeze_default_10: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_tensor_7, 0);  mul_tensor_7 = None
        unsqueeze_default_11: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_10, 2);  unsqueeze_default_10 = None
        unsqueeze_default_12: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_11, 3);  unsqueeze_default_11 = None
        mul_tensor_8: "f32[32, 192, 28, 28]" = torch.ops.aten.mul.Tensor(sub_tensor_1, unsqueeze_default_9);  sub_tensor_1 = unsqueeze_default_9 = None
        sub_tensor_2: "f32[32, 192, 28, 28]" = torch.ops.aten.sub.Tensor(getitem_242, mul_tensor_8);  getitem_242 = mul_tensor_8 = None
        sub_tensor_3: "f32[32, 192, 28, 28]" = torch.ops.aten.sub.Tensor(sub_tensor_2, unsqueeze_default_6);  sub_tensor_2 = unsqueeze_default_6 = None
        mul_tensor_9: "f32[32, 192, 28, 28]" = torch.ops.aten.mul.Tensor(sub_tensor_3, unsqueeze_default_12);  sub_tensor_3 = unsqueeze_default_12 = None
        add_tensor_2: "f32[32, 192, 28, 28]" = torch.ops.aten.add.Tensor(add_249, mul_tensor_9);  add_249 = mul_tensor_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/patch_embed.py:141 in forward, code: x = self.norm(x)
        squeeze_dims: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_3, [0, 2, 3]);  getitem_3 = None
        unsqueeze_default_13: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_dims, 0);  squeeze_dims = None
        unsqueeze_default_14: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_13, 2);  unsqueeze_default_13 = None
        unsqueeze_default_15: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_14, 3);  unsqueeze_default_14 = None
        sum_dim_int_list_2: "f32[192]" = torch.ops.aten.sum.dim_IntList(add_tensor_2, [0, 2, 3])
        sub_tensor_4: "f32[32, 192, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_default_15);  convolution_1 = unsqueeze_default_15 = None
        mul_tensor_10: "f32[32, 192, 28, 28]" = torch.ops.aten.mul.Tensor(add_tensor_2, sub_tensor_4)
        sum_dim_int_list_3: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_tensor_10, [0, 2, 3]);  mul_tensor_10 = None
        mul_tensor_11: "f32[192]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_2, 3.985969387755102e-05);  sum_dim_int_list_2 = None
        unsqueeze_default_16: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_tensor_11, 0);  mul_tensor_11 = None
        unsqueeze_default_17: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_16, 2);  unsqueeze_default_16 = None
        unsqueeze_default_18: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_17, 3);  unsqueeze_default_17 = None
        mul_tensor_12: "f32[192]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_3, 3.985969387755102e-05);  sum_dim_int_list_3 = None
        squeeze_dims_1: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_1, [0, 2, 3]);  rsqrt_1 = None
        mul_tensor_13: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_dims_1, squeeze_dims_1)
        mul_tensor_14: "f32[192]" = torch.ops.aten.mul.Tensor(mul_tensor_12, mul_tensor_13);  mul_tensor_12 = mul_tensor_13 = None
        unsqueeze_default_19: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_tensor_14, 0);  mul_tensor_14 = None
        unsqueeze_default_20: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_19, 2);  unsqueeze_default_19 = None
        unsqueeze_default_21: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_20, 3);  unsqueeze_default_20 = None
        mul_tensor_15: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_dims_1, primals_13);  squeeze_dims_1 = primals_13 = None
        unsqueeze_default_22: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_tensor_15, 0);  mul_tensor_15 = None
        unsqueeze_default_23: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_22, 2);  unsqueeze_default_22 = None
        unsqueeze_default_24: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_23, 3);  unsqueeze_default_23 = None
        mul_tensor_16: "f32[32, 192, 28, 28]" = torch.ops.aten.mul.Tensor(sub_tensor_4, unsqueeze_default_21);  sub_tensor_4 = unsqueeze_default_21 = None
        sub_tensor_5: "f32[32, 192, 28, 28]" = torch.ops.aten.sub.Tensor(add_tensor_2, mul_tensor_16);  add_tensor_2 = mul_tensor_16 = None
        sub_tensor_6: "f32[32, 192, 28, 28]" = torch.ops.aten.sub.Tensor(sub_tensor_5, unsqueeze_default_18);  sub_tensor_5 = unsqueeze_default_18 = None
        mul_tensor_17: "f32[32, 192, 28, 28]" = torch.ops.aten.mul.Tensor(sub_tensor_6, unsqueeze_default_24);  sub_tensor_6 = unsqueeze_default_24 = None
        return mul_tensor_17


def _default_make_inputs():
    return [
    torch.randn([32, 192, 28, 28], dtype=torch.float32, device='cuda'),
    torch.randn([32, 192, 28, 28], dtype=torch.float32, device='cuda'),
    torch.randn([1, 192, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 192, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([1, 192, 28, 28], dtype=torch.float32, device='cuda'),
    torch.randn([1, 192, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([32, 192, 28, 28], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
