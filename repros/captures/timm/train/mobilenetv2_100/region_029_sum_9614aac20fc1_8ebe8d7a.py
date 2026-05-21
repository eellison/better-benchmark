"""
Standalone repro captured via capture_hook.
Label: timm_mobilenetv2_100_train
Pattern hash: 9614aac20fc1
Shape hash: 8ebe8d7a
"""
_shapes_config = "(T([128, 24, 56, 56], f32, stride=(75264, 1, 1344, 24)), T([128, 24, 56, 56], f32, stride=(75264, 1, 1344, 24)), T([128, 24, 56, 56], f32, stride=(75264, 1, 1344, 24)), T([1, 24, 1, 1], f32), T([24], f32), T([24], f32))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, getitem_230: "f32[128, 24, 56, 56]", getitem_239: "f32[128, 24, 56, 56]", convolution_5: "f32[128, 24, 56, 56]", unsqueeze_762: "f32[1, 24, 1, 1]", squeeze_16: "f32[24]", primals_36: "f32[24]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:329 in forward, code: x = self.conv_pw(x)
        add_tensor: "f32[128, 24, 56, 56]" = torch.ops.aten.add.Tensor(getitem_230, getitem_239);  getitem_230 = getitem_239 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sum_dim_int_list: "f32[24]" = torch.ops.aten.sum.dim_IntList(add_tensor, [0, 2, 3])
        sub_tensor: "f32[128, 24, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_5, unsqueeze_762);  convolution_5 = unsqueeze_762 = None
        mul_tensor: "f32[128, 24, 56, 56]" = torch.ops.aten.mul.Tensor(add_tensor, sub_tensor)
        sum_dim_int_list_1: "f32[24]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [0, 2, 3]);  mul_tensor = None
        mul_tensor_1: "f32[24]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, 2.4912308673469386e-06);  sum_dim_int_list = None
        unsqueeze_default: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_tensor_1, 0);  mul_tensor_1 = None
        unsqueeze_default_1: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 2);  unsqueeze_default = None
        unsqueeze_default_2: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_1, 3);  unsqueeze_default_1 = None
        mul_tensor_2: "f32[24]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, 2.4912308673469386e-06);  sum_dim_int_list_1 = None
        mul_tensor_3: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_16, squeeze_16)
        mul_tensor_4: "f32[24]" = torch.ops.aten.mul.Tensor(mul_tensor_2, mul_tensor_3);  mul_tensor_2 = mul_tensor_3 = None
        unsqueeze_default_3: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_tensor_4, 0);  mul_tensor_4 = None
        unsqueeze_default_4: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_3, 2);  unsqueeze_default_3 = None
        unsqueeze_default_5: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 3);  unsqueeze_default_4 = None
        mul_tensor_5: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_16, primals_36);  squeeze_16 = primals_36 = None
        unsqueeze_default_6: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_tensor_5, 0);  mul_tensor_5 = None
        unsqueeze_default_7: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, 2);  unsqueeze_default_6 = None
        unsqueeze_default_8: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_7, 3);  unsqueeze_default_7 = None
        mul_tensor_6: "f32[128, 24, 56, 56]" = torch.ops.aten.mul.Tensor(sub_tensor, unsqueeze_default_5);  sub_tensor = unsqueeze_default_5 = None
        sub_tensor_1: "f32[128, 24, 56, 56]" = torch.ops.aten.sub.Tensor(add_tensor, mul_tensor_6);  add_tensor = mul_tensor_6 = None
        sub_tensor_2: "f32[128, 24, 56, 56]" = torch.ops.aten.sub.Tensor(sub_tensor_1, unsqueeze_default_2);  sub_tensor_1 = unsqueeze_default_2 = None
        mul_tensor_7: "f32[128, 24, 56, 56]" = torch.ops.aten.mul.Tensor(sub_tensor_2, unsqueeze_default_8);  sub_tensor_2 = unsqueeze_default_8 = None
        return mul_tensor_7



def make_inputs():
    return [
    torch.randn(9633792, dtype=torch.float32, device='cuda').as_strided([128, 24, 56, 56], [75264, 1, 1344, 24]),  # getitem_230
    torch.randn(9633792, dtype=torch.float32, device='cuda').as_strided([128, 24, 56, 56], [75264, 1, 1344, 24]),  # getitem_239
    torch.randn(9633792, dtype=torch.float32, device='cuda').as_strided([128, 24, 56, 56], [75264, 1, 1344, 24]),  # convolution_5
    torch.randn([1, 24, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([24], dtype=torch.float32, device='cuda'),
    torch.randn([24], dtype=torch.float32, device='cuda'),
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
