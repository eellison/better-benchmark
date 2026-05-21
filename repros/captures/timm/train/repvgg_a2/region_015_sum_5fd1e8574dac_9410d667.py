"""
Standalone repro captured via capture_hook.
Label: timm_repvgg_a2_train
Pattern hash: 5fd1e8574dac
Shape hash: 9410d667
"""
_shapes_config = "(T([128, 384, 14, 14], f32, stride=(75264, 1, 5376, 384)), T([128, 384, 14, 14], f32, stride=(75264, 1, 5376, 384)), T([128, 384, 14, 14], f32, stride=(75264, 1, 5376, 384)), T([], f32), T([128, 384, 14, 14], f32, stride=(75264, 1, 5376, 384)), T([1, 384, 1, 1], f32), T([384], f32), T([384], f32), T([128, 384, 14, 14], f32, stride=(75264, 1, 5376, 384)), T([1, 384, 1, 1], f32), T([384], f32), T([384], f32))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, getitem_122: "f32[128, 384, 14, 14]", getitem_125: "f32[128, 384, 14, 14]", relu_20: "f32[128, 384, 14, 14]", full_default: "f32[]", convolution_41: "f32[128, 384, 14, 14]", unsqueeze_270: "f32[1, 384, 1, 1]", squeeze_175: "f32[384]", primals_337: "f32[384]", convolution_40: "f32[128, 384, 14, 14]", unsqueeze_282: "f32[1, 384, 1, 1]", squeeze_172: "f32[384]", primals_331: "f32[384]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        add_tensor: "f32[128, 384, 14, 14]" = torch.ops.aten.add.Tensor(getitem_122, getitem_125);  getitem_122 = getitem_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/byobnet.py:755 in forward, code: return self.act(x)
        le_scalar: "b8[128, 384, 14, 14]" = torch.ops.aten.le.Scalar(relu_20, 0);  relu_20 = None
        where_self: "f32[128, 384, 14, 14]" = torch.ops.aten.where.self(le_scalar, full_default, add_tensor);  le_scalar = full_default = add_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sum_dim_int_list: "f32[384]" = torch.ops.aten.sum.dim_IntList(where_self, [0, 2, 3])
        sub_tensor: "f32[128, 384, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_41, unsqueeze_270);  convolution_41 = unsqueeze_270 = None
        mul_tensor: "f32[128, 384, 14, 14]" = torch.ops.aten.mul.Tensor(where_self, sub_tensor)
        sum_dim_int_list_1: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [0, 2, 3]);  mul_tensor = None
        mul_tensor_1: "f32[384]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, 3.985969387755102e-05);  sum_dim_int_list = None
        unsqueeze_default: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_tensor_1, 0);  mul_tensor_1 = None
        unsqueeze_default_1: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 2);  unsqueeze_default = None
        unsqueeze_default_2: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_1, 3);  unsqueeze_default_1 = None
        mul_tensor_2: "f32[384]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, 3.985969387755102e-05);  sum_dim_int_list_1 = None
        mul_tensor_3: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_175, squeeze_175)
        mul_tensor_4: "f32[384]" = torch.ops.aten.mul.Tensor(mul_tensor_2, mul_tensor_3);  mul_tensor_2 = mul_tensor_3 = None
        unsqueeze_default_3: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_tensor_4, 0);  mul_tensor_4 = None
        unsqueeze_default_4: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_3, 2);  unsqueeze_default_3 = None
        unsqueeze_default_5: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 3);  unsqueeze_default_4 = None
        mul_tensor_5: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_175, primals_337);  squeeze_175 = primals_337 = None
        unsqueeze_default_6: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_tensor_5, 0);  mul_tensor_5 = None
        unsqueeze_default_7: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, 2);  unsqueeze_default_6 = None
        unsqueeze_default_8: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_7, 3);  unsqueeze_default_7 = None
        mul_tensor_6: "f32[128, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_tensor, unsqueeze_default_5);  sub_tensor = unsqueeze_default_5 = None
        sub_tensor_1: "f32[128, 384, 14, 14]" = torch.ops.aten.sub.Tensor(where_self, mul_tensor_6);  mul_tensor_6 = None
        sub_tensor_2: "f32[128, 384, 14, 14]" = torch.ops.aten.sub.Tensor(sub_tensor_1, unsqueeze_default_2);  sub_tensor_1 = unsqueeze_default_2 = None
        mul_tensor_7: "f32[128, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_tensor_2, unsqueeze_default_8);  sub_tensor_2 = unsqueeze_default_8 = None
        sum_dim_int_list_2: "f32[384]" = torch.ops.aten.sum.dim_IntList(where_self, [0, 2, 3])
        sub_tensor_3: "f32[128, 384, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_40, unsqueeze_282);  convolution_40 = unsqueeze_282 = None
        mul_tensor_8: "f32[128, 384, 14, 14]" = torch.ops.aten.mul.Tensor(where_self, sub_tensor_3)
        sum_dim_int_list_3: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_tensor_8, [0, 2, 3]);  mul_tensor_8 = None
        mul_tensor_9: "f32[384]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_2, 3.985969387755102e-05);  sum_dim_int_list_2 = None
        unsqueeze_default_9: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_tensor_9, 0);  mul_tensor_9 = None
        unsqueeze_default_10: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_9, 2);  unsqueeze_default_9 = None
        unsqueeze_default_11: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_10, 3);  unsqueeze_default_10 = None
        mul_tensor_10: "f32[384]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_3, 3.985969387755102e-05);  sum_dim_int_list_3 = None
        mul_tensor_11: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_172, squeeze_172)
        mul_tensor_12: "f32[384]" = torch.ops.aten.mul.Tensor(mul_tensor_10, mul_tensor_11);  mul_tensor_10 = mul_tensor_11 = None
        unsqueeze_default_12: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_tensor_12, 0);  mul_tensor_12 = None
        unsqueeze_default_13: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_12, 2);  unsqueeze_default_12 = None
        unsqueeze_default_14: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_13, 3);  unsqueeze_default_13 = None
        mul_tensor_13: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_172, primals_331);  squeeze_172 = primals_331 = None
        unsqueeze_default_15: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_tensor_13, 0);  mul_tensor_13 = None
        unsqueeze_default_16: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_15, 2);  unsqueeze_default_15 = None
        unsqueeze_default_17: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_16, 3);  unsqueeze_default_16 = None
        mul_tensor_14: "f32[128, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_tensor_3, unsqueeze_default_14);  sub_tensor_3 = unsqueeze_default_14 = None
        sub_tensor_4: "f32[128, 384, 14, 14]" = torch.ops.aten.sub.Tensor(where_self, mul_tensor_14);  where_self = mul_tensor_14 = None
        sub_tensor_5: "f32[128, 384, 14, 14]" = torch.ops.aten.sub.Tensor(sub_tensor_4, unsqueeze_default_11);  sub_tensor_4 = unsqueeze_default_11 = None
        mul_tensor_15: "f32[128, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_tensor_5, unsqueeze_default_17);  sub_tensor_5 = unsqueeze_default_17 = None
        return (mul_tensor_7, mul_tensor_15)



def make_inputs():
    return [
    torch.randn(9633792, dtype=torch.float32, device='cuda').as_strided([128, 384, 14, 14], [75264, 1, 5376, 384]),  # getitem_122
    torch.randn(9633792, dtype=torch.float32, device='cuda').as_strided([128, 384, 14, 14], [75264, 1, 5376, 384]),  # getitem_125
    torch.randn(9633792, dtype=torch.float32, device='cuda').as_strided([128, 384, 14, 14], [75264, 1, 5376, 384]),  # relu_20
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn(9633792, dtype=torch.float32, device='cuda').as_strided([128, 384, 14, 14], [75264, 1, 5376, 384]),  # convolution_41
    torch.randn([1, 384, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn(9633792, dtype=torch.float32, device='cuda').as_strided([128, 384, 14, 14], [75264, 1, 5376, 384]),  # convolution_40
    torch.randn([1, 384, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
