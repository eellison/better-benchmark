"""
Standalone repro captured via capture_hook.
Label: resnet18_training
Pattern hash: c65e380ac58f
Shape hash: e6cea75b
"""
import torch
import torch._inductor.config as inductor_config
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, where_6: "f32[4, 256, 14, 14]", convolution_12: "f32[4, 256, 14, 14]", unsqueeze_166: "f32[1, 256, 1, 1]", squeeze_37: "f32[256]", primals_78: "f32[256]", relu_9: "f32[4, 256, 14, 14]", full_default: "f32[]", getitem_66: "f32[4, 256, 14, 14]", convolution_10: "f32[4, 256, 14, 14]", unsqueeze_190: "f32[1, 256, 1, 1]", squeeze_31: "f32[256]", primals_66: "f32[256]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:100 in forward, code: identity = self.downsample(x)
        sum_dim_int_list: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_6, [0, 2, 3])
        sub_tensor: "f32[4, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_12, unsqueeze_166);  convolution_12 = unsqueeze_166 = None
        mul_tensor: "f32[4, 256, 14, 14]" = torch.ops.aten.mul.Tensor(where_6, sub_tensor)
        sum_dim_int_list_1: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [0, 2, 3]);  mul_tensor = None
        mul_tensor_1: "f32[256]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, 0.0012755102040816326);  sum_dim_int_list = None
        unsqueeze_default: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_tensor_1, 0);  mul_tensor_1 = None
        unsqueeze_default_1: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 2);  unsqueeze_default = None
        unsqueeze_default_2: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_1, 3);  unsqueeze_default_1 = None
        mul_tensor_2: "f32[256]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, 0.0012755102040816326);  sum_dim_int_list_1 = None
        mul_tensor_3: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_37, squeeze_37)
        mul_tensor_4: "f32[256]" = torch.ops.aten.mul.Tensor(mul_tensor_2, mul_tensor_3);  mul_tensor_2 = mul_tensor_3 = None
        unsqueeze_default_3: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_tensor_4, 0);  mul_tensor_4 = None
        unsqueeze_default_4: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_3, 2);  unsqueeze_default_3 = None
        unsqueeze_default_5: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 3);  unsqueeze_default_4 = None
        mul_tensor_5: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_37, primals_78);  squeeze_37 = primals_78 = None
        unsqueeze_default_6: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_tensor_5, 0);  mul_tensor_5 = None
        unsqueeze_default_7: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, 2);  unsqueeze_default_6 = None
        unsqueeze_default_8: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_7, 3);  unsqueeze_default_7 = None
        mul_tensor_6: "f32[4, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_tensor, unsqueeze_default_5);  sub_tensor = unsqueeze_default_5 = None
        sub_tensor_1: "f32[4, 256, 14, 14]" = torch.ops.aten.sub.Tensor(where_6, mul_tensor_6);  where_6 = mul_tensor_6 = None
        sub_tensor_2: "f32[4, 256, 14, 14]" = torch.ops.aten.sub.Tensor(sub_tensor_1, unsqueeze_default_2);  sub_tensor_1 = unsqueeze_default_2 = None
        mul_tensor_7: "f32[4, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_tensor_2, unsqueeze_default_8);  sub_tensor_2 = unsqueeze_default_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:94 in forward, code: out = self.relu(out)
        le_scalar: "b8[4, 256, 14, 14]" = torch.ops.aten.le.Scalar(relu_9, 0);  relu_9 = None
        where_self: "f32[4, 256, 14, 14]" = torch.ops.aten.where.self(le_scalar, full_default, getitem_66);  le_scalar = full_default = getitem_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:93 in forward, code: out = self.bn1(out)
        sum_dim_int_list_2: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_self, [0, 2, 3])
        sub_tensor_3: "f32[4, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_10, unsqueeze_190);  convolution_10 = unsqueeze_190 = None
        mul_tensor_8: "f32[4, 256, 14, 14]" = torch.ops.aten.mul.Tensor(where_self, sub_tensor_3)
        sum_dim_int_list_3: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_tensor_8, [0, 2, 3]);  mul_tensor_8 = None
        mul_tensor_9: "f32[256]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_2, 0.0012755102040816326);  sum_dim_int_list_2 = None
        unsqueeze_default_9: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_tensor_9, 0);  mul_tensor_9 = None
        unsqueeze_default_10: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_9, 2);  unsqueeze_default_9 = None
        unsqueeze_default_11: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_10, 3);  unsqueeze_default_10 = None
        mul_tensor_10: "f32[256]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_3, 0.0012755102040816326);  sum_dim_int_list_3 = None
        mul_tensor_11: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_31, squeeze_31)
        mul_tensor_12: "f32[256]" = torch.ops.aten.mul.Tensor(mul_tensor_10, mul_tensor_11);  mul_tensor_10 = mul_tensor_11 = None
        unsqueeze_default_12: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_tensor_12, 0);  mul_tensor_12 = None
        unsqueeze_default_13: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_12, 2);  unsqueeze_default_12 = None
        unsqueeze_default_14: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_13, 3);  unsqueeze_default_13 = None
        mul_tensor_13: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_31, primals_66);  squeeze_31 = primals_66 = None
        unsqueeze_default_15: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_tensor_13, 0);  mul_tensor_13 = None
        unsqueeze_default_16: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_15, 2);  unsqueeze_default_15 = None
        unsqueeze_default_17: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_16, 3);  unsqueeze_default_16 = None
        mul_tensor_14: "f32[4, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_tensor_3, unsqueeze_default_14);  sub_tensor_3 = unsqueeze_default_14 = None
        sub_tensor_4: "f32[4, 256, 14, 14]" = torch.ops.aten.sub.Tensor(where_self, mul_tensor_14);  where_self = mul_tensor_14 = None
        sub_tensor_5: "f32[4, 256, 14, 14]" = torch.ops.aten.sub.Tensor(sub_tensor_4, unsqueeze_default_11);  sub_tensor_4 = unsqueeze_default_11 = None
        mul_tensor_15: "f32[4, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_tensor_5, unsqueeze_default_17);  sub_tensor_5 = unsqueeze_default_17 = None
        return (mul_tensor_7, mul_tensor_15)



def make_inputs():
    return [
    torch.randn(200704, dtype=torch.float32, device='cuda').as_strided([4, 256, 14, 14], [50176, 1, 3584, 256]),  # where_6
    torch.randn(200704, dtype=torch.float32, device='cuda').as_strided([4, 256, 14, 14], [50176, 1, 3584, 256]),  # convolution_12
    torch.randn([1, 256, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([256], dtype=torch.float32, device='cuda'),
    torch.randn([256], dtype=torch.float32, device='cuda'),
    torch.randn(200704, dtype=torch.float32, device='cuda').as_strided([4, 256, 14, 14], [50176, 1, 3584, 256]),  # relu_9
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn(200704, dtype=torch.float32, device='cuda').as_strided([4, 256, 14, 14], [50176, 1, 3584, 256]),  # getitem_66
    torch.randn(200704, dtype=torch.float32, device='cuda').as_strided([4, 256, 14, 14], [50176, 1, 3584, 256]),  # convolution_10
    torch.randn([1, 256, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([256], dtype=torch.float32, device='cuda'),
    torch.randn([256], dtype=torch.float32, device='cuda'),
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
