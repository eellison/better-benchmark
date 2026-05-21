"""
Standalone repro captured via capture_hook.
Label: timm_visformer_small_train
Pattern hash: 944fbe410b44
Shape hash: 5f09b7f8
"""
_shapes_config = "(T([128, 384, 14, 14], f32, stride=(75264, 1, 5376, 384)), T([128, 384, 14, 14], f32, stride=(75264, 1, 5376, 384)), T([1, 384, 1, 1], f32), T([1, 384, 1, 1], f32), T([384], f32), T([384], f32), T([1, 384, 14, 14], f32, stride=(75264, 1, 5376, 384)), T([1, 384, 1, 1], f32), T([384], f32), T([384], f32), T([128, 384, 14, 14], f32, stride=(75264, 1, 5376, 384)))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, getitem_176: "f32[128, 384, 14, 14]", convolution_23: "f32[128, 384, 14, 14]", getitem_19: "f32[1, 384, 1, 1]", rsqrt_9: "f32[1, 384, 1, 1]", primals_77: "f32[384]", primals_78: "f32[384]", primals_79: "f32[1, 384, 14, 14]", unsqueeze_318: "f32[1, 384, 1, 1]", squeeze_31: "f32[384]", primals_83: "f32[384]", add_218: "f32[128, 384, 14, 14]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:176 in forward, code: x = x + self.drop_path(self.attn(self.norm1(x)))
        sum_dim_int_list: "f32[384]" = torch.ops.aten.sum.dim_IntList(getitem_176, [0, 2, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/patch_embed.py:141 in forward, code: x = self.norm(x)
        sub_tensor: "f32[128, 384, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_23, getitem_19)
        mul_tensor: "f32[128, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_9);  sub_tensor = None
        unsqueeze_default: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_77, -1)
        unsqueeze_default_1: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_1: "f32[128, 384, 14, 14]" = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default_1);  mul_tensor = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_78, -1);  primals_78 = None
        unsqueeze_default_3: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor: "f32[128, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_tensor_1, unsqueeze_default_3);  mul_tensor_1 = unsqueeze_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:452 in forward_features, code: x = self.pos_drop(x + self.pos_embed2)
        add_tensor_1: "f32[128, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_tensor, primals_79);  add_tensor = primals_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:176 in forward, code: x = x + self.drop_path(self.attn(self.norm1(x)))
        sub_tensor_1: "f32[128, 384, 14, 14]" = torch.ops.aten.sub.Tensor(add_tensor_1, unsqueeze_318);  add_tensor_1 = unsqueeze_318 = None
        mul_tensor_2: "f32[128, 384, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_176, sub_tensor_1)
        sum_dim_int_list_1: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [0, 2, 3]);  mul_tensor_2 = None
        mul_tensor_3: "f32[384]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, 3.985969387755102e-05);  sum_dim_int_list = None
        unsqueeze_default_4: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_tensor_3, 0);  mul_tensor_3 = None
        unsqueeze_default_5: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 2);  unsqueeze_default_4 = None
        unsqueeze_default_6: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_5, 3);  unsqueeze_default_5 = None
        mul_tensor_4: "f32[384]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, 3.985969387755102e-05);  sum_dim_int_list_1 = None
        mul_tensor_5: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_31, squeeze_31)
        mul_tensor_6: "f32[384]" = torch.ops.aten.mul.Tensor(mul_tensor_4, mul_tensor_5);  mul_tensor_4 = mul_tensor_5 = None
        unsqueeze_default_7: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_tensor_6, 0);  mul_tensor_6 = None
        unsqueeze_default_8: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_7, 2);  unsqueeze_default_7 = None
        unsqueeze_default_9: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_8, 3);  unsqueeze_default_8 = None
        mul_tensor_7: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_31, primals_83);  squeeze_31 = primals_83 = None
        unsqueeze_default_10: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_tensor_7, 0);  mul_tensor_7 = None
        unsqueeze_default_11: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_10, 2);  unsqueeze_default_10 = None
        unsqueeze_default_12: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_11, 3);  unsqueeze_default_11 = None
        mul_tensor_8: "f32[128, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_tensor_1, unsqueeze_default_9);  sub_tensor_1 = unsqueeze_default_9 = None
        sub_tensor_2: "f32[128, 384, 14, 14]" = torch.ops.aten.sub.Tensor(getitem_176, mul_tensor_8);  getitem_176 = mul_tensor_8 = None
        sub_tensor_3: "f32[128, 384, 14, 14]" = torch.ops.aten.sub.Tensor(sub_tensor_2, unsqueeze_default_6);  sub_tensor_2 = unsqueeze_default_6 = None
        mul_tensor_9: "f32[128, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_tensor_3, unsqueeze_default_12);  sub_tensor_3 = unsqueeze_default_12 = None
        add_tensor_2: "f32[128, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_218, mul_tensor_9);  add_218 = mul_tensor_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/patch_embed.py:141 in forward, code: x = self.norm(x)
        squeeze_dims: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_19, [0, 2, 3]);  getitem_19 = None
        unsqueeze_default_13: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_dims, 0);  squeeze_dims = None
        unsqueeze_default_14: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_13, 2);  unsqueeze_default_13 = None
        unsqueeze_default_15: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_14, 3);  unsqueeze_default_14 = None
        sum_dim_int_list_2: "f32[384]" = torch.ops.aten.sum.dim_IntList(add_tensor_2, [0, 2, 3])
        sub_tensor_4: "f32[128, 384, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_23, unsqueeze_default_15);  convolution_23 = unsqueeze_default_15 = None
        mul_tensor_10: "f32[128, 384, 14, 14]" = torch.ops.aten.mul.Tensor(add_tensor_2, sub_tensor_4)
        sum_dim_int_list_3: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_tensor_10, [0, 2, 3]);  mul_tensor_10 = None
        mul_tensor_11: "f32[384]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_2, 3.985969387755102e-05);  sum_dim_int_list_2 = None
        unsqueeze_default_16: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_tensor_11, 0);  mul_tensor_11 = None
        unsqueeze_default_17: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_16, 2);  unsqueeze_default_16 = None
        unsqueeze_default_18: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_17, 3);  unsqueeze_default_17 = None
        mul_tensor_12: "f32[384]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_3, 3.985969387755102e-05);  sum_dim_int_list_3 = None
        squeeze_dims_1: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_9, [0, 2, 3]);  rsqrt_9 = None
        mul_tensor_13: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_1, squeeze_dims_1)
        mul_tensor_14: "f32[384]" = torch.ops.aten.mul.Tensor(mul_tensor_12, mul_tensor_13);  mul_tensor_12 = mul_tensor_13 = None
        unsqueeze_default_19: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_tensor_14, 0);  mul_tensor_14 = None
        unsqueeze_default_20: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_19, 2);  unsqueeze_default_19 = None
        unsqueeze_default_21: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_20, 3);  unsqueeze_default_20 = None
        mul_tensor_15: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_1, primals_77);  squeeze_dims_1 = primals_77 = None
        unsqueeze_default_22: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_tensor_15, 0);  mul_tensor_15 = None
        unsqueeze_default_23: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_22, 2);  unsqueeze_default_22 = None
        unsqueeze_default_24: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_23, 3);  unsqueeze_default_23 = None
        mul_tensor_16: "f32[128, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_tensor_4, unsqueeze_default_21);  sub_tensor_4 = unsqueeze_default_21 = None
        sub_tensor_5: "f32[128, 384, 14, 14]" = torch.ops.aten.sub.Tensor(add_tensor_2, mul_tensor_16);  add_tensor_2 = mul_tensor_16 = None
        sub_tensor_6: "f32[128, 384, 14, 14]" = torch.ops.aten.sub.Tensor(sub_tensor_5, unsqueeze_default_18);  sub_tensor_5 = unsqueeze_default_18 = None
        mul_tensor_17: "f32[128, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_tensor_6, unsqueeze_default_24);  sub_tensor_6 = unsqueeze_default_24 = None
        return mul_tensor_17



def make_inputs():
    return [
    torch.randn(9633792, dtype=torch.float32, device='cuda').as_strided([128, 384, 14, 14], [75264, 1, 5376, 384]),  # getitem_176
    torch.randn(9633792, dtype=torch.float32, device='cuda').as_strided([128, 384, 14, 14], [75264, 1, 5376, 384]),  # convolution_23
    torch.randn([1, 384, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 384, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn(75264, dtype=torch.float32, device='cuda').as_strided([1, 384, 14, 14], [75264, 1, 5376, 384]),  # primals_79
    torch.randn([1, 384, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn(9633792, dtype=torch.float32, device='cuda').as_strided([128, 384, 14, 14], [75264, 1, 5376, 384]),  # add_218
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
