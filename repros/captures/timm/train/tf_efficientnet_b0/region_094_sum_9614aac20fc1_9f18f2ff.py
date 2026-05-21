"""
Standalone repro captured via capture_hook.
Label: timm_tf_efficientnet_b0_train
Pattern hash: 9614aac20fc1
Shape hash: 9f18f2ff
"""
_shapes_config = "(T([128, 192, 7, 7], f32, stride=(9408, 1, 1344, 192)), T([128, 192, 7, 7], f32, stride=(9408, 1, 1344, 192)), T([128, 192, 7, 7], f32, stride=(9408, 1, 1344, 192)), T([1, 192, 1, 1], f32), T([192], f32), T([192], f32))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, add_327: "f32[128, 192, 7, 7]", getitem_158: "f32[128, 192, 7, 7]", convolution_59: "f32[128, 192, 7, 7]", unsqueeze_354: "f32[1, 192, 1, 1]", squeeze_106: "f32[192]", primals_264: "f32[192]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:329 in forward, code: x = self.conv_pw(x)
        add_tensor: "f32[128, 192, 7, 7]" = torch.ops.aten.add.Tensor(add_327, getitem_158);  add_327 = getitem_158 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sum_dim_int_list: "f32[192]" = torch.ops.aten.sum.dim_IntList(add_tensor, [0, 2, 3])
        sub_tensor: "f32[128, 192, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_59, unsqueeze_354);  convolution_59 = unsqueeze_354 = None
        mul_tensor: "f32[128, 192, 7, 7]" = torch.ops.aten.mul.Tensor(add_tensor, sub_tensor)
        sum_dim_int_list_1: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [0, 2, 3]);  mul_tensor = None
        mul_tensor_1: "f32[192]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, 0.00015943877551020407);  sum_dim_int_list = None
        unsqueeze_default: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_tensor_1, 0);  mul_tensor_1 = None
        unsqueeze_default_1: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 2);  unsqueeze_default = None
        unsqueeze_default_2: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_1, 3);  unsqueeze_default_1 = None
        mul_tensor_2: "f32[192]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, 0.00015943877551020407);  sum_dim_int_list_1 = None
        mul_tensor_3: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_106, squeeze_106)
        mul_tensor_4: "f32[192]" = torch.ops.aten.mul.Tensor(mul_tensor_2, mul_tensor_3);  mul_tensor_2 = mul_tensor_3 = None
        unsqueeze_default_3: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_tensor_4, 0);  mul_tensor_4 = None
        unsqueeze_default_4: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_3, 2);  unsqueeze_default_3 = None
        unsqueeze_default_5: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 3);  unsqueeze_default_4 = None
        mul_tensor_5: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_106, primals_264);  squeeze_106 = primals_264 = None
        unsqueeze_default_6: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_tensor_5, 0);  mul_tensor_5 = None
        unsqueeze_default_7: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, 2);  unsqueeze_default_6 = None
        unsqueeze_default_8: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_7, 3);  unsqueeze_default_7 = None
        mul_tensor_6: "f32[128, 192, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor, unsqueeze_default_5);  sub_tensor = unsqueeze_default_5 = None
        sub_tensor_1: "f32[128, 192, 7, 7]" = torch.ops.aten.sub.Tensor(add_tensor, mul_tensor_6);  add_tensor = mul_tensor_6 = None
        sub_tensor_2: "f32[128, 192, 7, 7]" = torch.ops.aten.sub.Tensor(sub_tensor_1, unsqueeze_default_2);  sub_tensor_1 = unsqueeze_default_2 = None
        mul_tensor_7: "f32[128, 192, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor_2, unsqueeze_default_8);  sub_tensor_2 = unsqueeze_default_8 = None
        return mul_tensor_7



def make_inputs():
    return [
    torch.randn(1204224, dtype=torch.float32, device='cuda').as_strided([128, 192, 7, 7], [9408, 1, 1344, 192]),  # add_327
    torch.randn(1204224, dtype=torch.float32, device='cuda').as_strided([128, 192, 7, 7], [9408, 1, 1344, 192]),  # getitem_158
    torch.randn(1204224, dtype=torch.float32, device='cuda').as_strided([128, 192, 7, 7], [9408, 1, 1344, 192]),  # convolution_59
    torch.randn([1, 192, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
