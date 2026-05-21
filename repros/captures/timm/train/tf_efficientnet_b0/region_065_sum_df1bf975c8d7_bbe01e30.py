"""
Standalone repro captured via capture_hook.
Label: timm_tf_efficientnet_b0_train
Pattern hash: df1bf975c8d7
Shape hash: bbe01e30
"""
_shapes_config = "(T([128, 144, 56, 56], f32, stride=(451584, 1, 8064, 144)), T([1, 144, 1, 1], f32), T([1, 144, 1, 1], f32), T([144], f32), T([144], f32), T([128, 144, 56, 56], f32, stride=(451584, 1, 8064, 144)), T([128, 144, 1, 1], f32))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, convolution_11: "f32[128, 144, 56, 56]", getitem_15: "f32[1, 144, 1, 1]", rsqrt_7: "f32[1, 144, 1, 1]", primals_56: "f32[144]", primals_57: "f32[144]", getitem_296: "f32[128, 144, 56, 56]", convolution_13: "f32[128, 144, 1, 1]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_tensor: "f32[128, 144, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_11, getitem_15);  convolution_11 = getitem_15 = None
        mul_tensor: "f32[128, 144, 56, 56]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_7);  sub_tensor = rsqrt_7 = None
        unsqueeze_default: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(primals_56, -1);  primals_56 = None
        unsqueeze_default_1: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_1: "f32[128, 144, 56, 56]" = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default_1);  mul_tensor = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(primals_57, -1);  primals_57 = None
        unsqueeze_default_3: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor: "f32[128, 144, 56, 56]" = torch.ops.aten.add.Tensor(mul_tensor_1, unsqueeze_default_3);  mul_tensor_1 = unsqueeze_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        neg_default: "f32[128, 144, 56, 56]" = torch.ops.aten.neg.default(add_tensor)
        exp_default: "f32[128, 144, 56, 56]" = torch.ops.aten.exp.default(neg_default);  neg_default = None
        add_tensor_1: "f32[128, 144, 56, 56]" = torch.ops.aten.add.Tensor(exp_default, 1);  exp_default = None
        div_tensor: "f32[128, 144, 56, 56]" = torch.ops.aten.div.Tensor(add_tensor, add_tensor_1);  add_tensor = add_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:83 in forward, code: return x * self.gate(x_se)
        mul_tensor_2: "f32[128, 144, 56, 56]" = torch.ops.aten.mul.Tensor(getitem_296, div_tensor);  getitem_296 = div_tensor = None
        sigmoid_default: "f32[128, 144, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_13);  convolution_13 = None
        sum_dim_int_list: "f32[128, 144, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [2, 3], True);  mul_tensor_2 = None
        sub_tensor_1: "f32[128, 144, 1, 1]" = torch.ops.aten.sub.Tensor(1, sigmoid_default)
        mul_tensor_3: "f32[128, 144, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_default, sub_tensor_1);  sigmoid_default = sub_tensor_1 = None
        mul_tensor_4: "f32[128, 144, 1, 1]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, mul_tensor_3);  sum_dim_int_list = mul_tensor_3 = None
        return mul_tensor_4



def make_inputs():
    return [
    torch.randn(57802752, dtype=torch.float32, device='cuda').as_strided([128, 144, 56, 56], [451584, 1, 8064, 144]),  # convolution_11
    torch.randn([1, 144, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 144, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([144], dtype=torch.float32, device='cuda'),
    torch.randn([144], dtype=torch.float32, device='cuda'),
    torch.randn(57802752, dtype=torch.float32, device='cuda').as_strided([128, 144, 56, 56], [451584, 1, 8064, 144]),  # getitem_296
    torch.randn([128, 144, 1, 1], dtype=torch.float32, device='cuda'),
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
