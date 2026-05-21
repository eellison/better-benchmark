"""
Standalone repro captured via capture_hook.
Label: timm_mobilevit_s_train
Pattern hash: c96550c0cb51
Shape hash: 232da079
"""
_shapes_config = "(T([128, 32, 128, 128], f32, stride=(524288, 1, 4096, 32)), T([128, 32, 128, 128], f32, stride=(524288, 1, 4096, 32)), T([1, 32, 1, 1], f32), T([32], f32), T([32], f32))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, getitem_295: "f32[128, 32, 128, 128]", convolution_3: "f32[128, 32, 128, 128]", unsqueeze_466: "f32[1, 32, 1, 1]", squeeze_10: "f32[32]", primals_24: "f32[32]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sum_dim_int_list: "f32[32]" = torch.ops.aten.sum.dim_IntList(getitem_295, [0, 2, 3])
        sub_tensor: "f32[128, 32, 128, 128]" = torch.ops.aten.sub.Tensor(convolution_3, unsqueeze_466);  convolution_3 = unsqueeze_466 = None
        mul_tensor: "f32[128, 32, 128, 128]" = torch.ops.aten.mul.Tensor(getitem_295, sub_tensor)
        sum_dim_int_list_1: "f32[32]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [0, 2, 3]);  mul_tensor = None
        mul_tensor_1: "f32[32]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, 4.76837158203125e-07);  sum_dim_int_list = None
        unsqueeze_default: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_tensor_1, 0);  mul_tensor_1 = None
        unsqueeze_default_1: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 2);  unsqueeze_default = None
        unsqueeze_default_2: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_1, 3);  unsqueeze_default_1 = None
        mul_tensor_2: "f32[32]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, 4.76837158203125e-07);  sum_dim_int_list_1 = None
        mul_tensor_3: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_10, squeeze_10)
        mul_tensor_4: "f32[32]" = torch.ops.aten.mul.Tensor(mul_tensor_2, mul_tensor_3);  mul_tensor_2 = mul_tensor_3 = None
        unsqueeze_default_3: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_tensor_4, 0);  mul_tensor_4 = None
        unsqueeze_default_4: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_3, 2);  unsqueeze_default_3 = None
        unsqueeze_default_5: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 3);  unsqueeze_default_4 = None
        mul_tensor_5: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_10, primals_24);  squeeze_10 = primals_24 = None
        unsqueeze_default_6: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_tensor_5, 0);  mul_tensor_5 = None
        unsqueeze_default_7: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, 2);  unsqueeze_default_6 = None
        unsqueeze_default_8: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_7, 3);  unsqueeze_default_7 = None
        mul_tensor_6: "f32[128, 32, 128, 128]" = torch.ops.aten.mul.Tensor(sub_tensor, unsqueeze_default_5);  sub_tensor = unsqueeze_default_5 = None
        sub_tensor_1: "f32[128, 32, 128, 128]" = torch.ops.aten.sub.Tensor(getitem_295, mul_tensor_6);  getitem_295 = mul_tensor_6 = None
        sub_tensor_2: "f32[128, 32, 128, 128]" = torch.ops.aten.sub.Tensor(sub_tensor_1, unsqueeze_default_2);  sub_tensor_1 = unsqueeze_default_2 = None
        mul_tensor_7: "f32[128, 32, 128, 128]" = torch.ops.aten.mul.Tensor(sub_tensor_2, unsqueeze_default_8);  sub_tensor_2 = unsqueeze_default_8 = None
        return mul_tensor_7



def make_inputs():
    return [
    torch.randn(67108864, dtype=torch.float32, device='cuda').as_strided([128, 32, 128, 128], [524288, 1, 4096, 32]),  # getitem_295
    torch.randn(67108864, dtype=torch.float32, device='cuda').as_strided([128, 32, 128, 128], [524288, 1, 4096, 32]),  # convolution_3
    torch.randn([1, 32, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32], dtype=torch.float32, device='cuda'),
    torch.randn([32], dtype=torch.float32, device='cuda'),
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
