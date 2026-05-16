"""
Standalone repro captured via capture_hook.
Label: wide_resnet50_2_training
Pattern hash: c65e380ac58f
Shape hash: 9b1ba7cc
"""
import torch
import torch._inductor.config as inductor_config
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, where_24: "f32[4, 1024, 14, 14]", convolution_27: "f32[4, 1024, 14, 14]", unsqueeze_514: "f32[1, 1024, 1, 1]", squeeze_82: "f32[1024]", primals_168: "f32[1024]", relu_22: "f32[4, 512, 28, 28]", full_default: "f32[]", getitem_189: "f32[4, 512, 28, 28]", convolution_24: "f32[4, 512, 28, 28]", unsqueeze_550: "f32[1, 512, 1, 1]", squeeze_73: "f32[512]", primals_150: "f32[512]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:158 in forward, code: identity = self.downsample(x)
        sum_dim_int_list: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_24, [0, 2, 3])
        sub_tensor: "f32[4, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_27, unsqueeze_514);  convolution_27 = unsqueeze_514 = None
        mul_tensor: "f32[4, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(where_24, sub_tensor)
        sum_dim_int_list_1: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [0, 2, 3]);  mul_tensor = None
        mul_tensor_1: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, 0.0012755102040816326);  sum_dim_int_list = None
        unsqueeze_default: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_tensor_1, 0);  mul_tensor_1 = None
        unsqueeze_default_1: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 2);  unsqueeze_default = None
        unsqueeze_default_2: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_1, 3);  unsqueeze_default_1 = None
        mul_tensor_2: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, 0.0012755102040816326);  sum_dim_int_list_1 = None
        mul_tensor_3: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_82, squeeze_82)
        mul_tensor_4: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_tensor_2, mul_tensor_3);  mul_tensor_2 = mul_tensor_3 = None
        unsqueeze_default_3: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_tensor_4, 0);  mul_tensor_4 = None
        unsqueeze_default_4: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_3, 2);  unsqueeze_default_3 = None
        unsqueeze_default_5: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 3);  unsqueeze_default_4 = None
        mul_tensor_5: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_82, primals_168);  squeeze_82 = primals_168 = None
        unsqueeze_default_6: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_tensor_5, 0);  mul_tensor_5 = None
        unsqueeze_default_7: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, 2);  unsqueeze_default_6 = None
        unsqueeze_default_8: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_7, 3);  unsqueeze_default_7 = None
        mul_tensor_6: "f32[4, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_tensor, unsqueeze_default_5);  sub_tensor = unsqueeze_default_5 = None
        sub_tensor_1: "f32[4, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(where_24, mul_tensor_6);  where_24 = mul_tensor_6 = None
        sub_tensor_2: "f32[4, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(sub_tensor_1, unsqueeze_default_2);  sub_tensor_1 = unsqueeze_default_2 = None
        mul_tensor_7: "f32[4, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_tensor_2, unsqueeze_default_8);  sub_tensor_2 = unsqueeze_default_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        le_scalar: "b8[4, 512, 28, 28]" = torch.ops.aten.le.Scalar(relu_22, 0);  relu_22 = None
        where_self: "f32[4, 512, 28, 28]" = torch.ops.aten.where.self(le_scalar, full_default, getitem_189);  le_scalar = full_default = getitem_189 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        sum_dim_int_list_2: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_self, [0, 2, 3])
        sub_tensor_3: "f32[4, 512, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_24, unsqueeze_550);  convolution_24 = unsqueeze_550 = None
        mul_tensor_8: "f32[4, 512, 28, 28]" = torch.ops.aten.mul.Tensor(where_self, sub_tensor_3)
        sum_dim_int_list_3: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_8, [0, 2, 3]);  mul_tensor_8 = None
        mul_tensor_9: "f32[512]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_2, 0.00031887755102040814);  sum_dim_int_list_2 = None
        unsqueeze_default_9: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_tensor_9, 0);  mul_tensor_9 = None
        unsqueeze_default_10: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_9, 2);  unsqueeze_default_9 = None
        unsqueeze_default_11: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_10, 3);  unsqueeze_default_10 = None
        mul_tensor_10: "f32[512]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_3, 0.00031887755102040814);  sum_dim_int_list_3 = None
        mul_tensor_11: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_73, squeeze_73)
        mul_tensor_12: "f32[512]" = torch.ops.aten.mul.Tensor(mul_tensor_10, mul_tensor_11);  mul_tensor_10 = mul_tensor_11 = None
        unsqueeze_default_12: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_tensor_12, 0);  mul_tensor_12 = None
        unsqueeze_default_13: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_12, 2);  unsqueeze_default_12 = None
        unsqueeze_default_14: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_13, 3);  unsqueeze_default_13 = None
        mul_tensor_13: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_73, primals_150);  squeeze_73 = primals_150 = None
        unsqueeze_default_15: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_tensor_13, 0);  mul_tensor_13 = None
        unsqueeze_default_16: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_15, 2);  unsqueeze_default_15 = None
        unsqueeze_default_17: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_16, 3);  unsqueeze_default_16 = None
        mul_tensor_14: "f32[4, 512, 28, 28]" = torch.ops.aten.mul.Tensor(sub_tensor_3, unsqueeze_default_14);  sub_tensor_3 = unsqueeze_default_14 = None
        sub_tensor_4: "f32[4, 512, 28, 28]" = torch.ops.aten.sub.Tensor(where_self, mul_tensor_14);  where_self = mul_tensor_14 = None
        sub_tensor_5: "f32[4, 512, 28, 28]" = torch.ops.aten.sub.Tensor(sub_tensor_4, unsqueeze_default_11);  sub_tensor_4 = unsqueeze_default_11 = None
        mul_tensor_15: "f32[4, 512, 28, 28]" = torch.ops.aten.mul.Tensor(sub_tensor_5, unsqueeze_default_17);  sub_tensor_5 = unsqueeze_default_17 = None
        return (mul_tensor_7, mul_tensor_15)



def make_inputs():
    return [
    torch.randn(802816, dtype=torch.float32, device='cuda').as_strided([4, 1024, 14, 14], [200704, 1, 14336, 1024]),  # where_24
    torch.randn(802816, dtype=torch.float32, device='cuda').as_strided([4, 1024, 14, 14], [200704, 1, 14336, 1024]),  # convolution_27
    torch.randn([1, 1024, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1024], dtype=torch.float32, device='cuda'),
    torch.randn([1024], dtype=torch.float32, device='cuda'),
    torch.randn(1605632, dtype=torch.float32, device='cuda').as_strided([4, 512, 28, 28], [401408, 1, 14336, 512]),  # relu_22
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn(1605632, dtype=torch.float32, device='cuda').as_strided([4, 512, 28, 28], [401408, 1, 14336, 512]),  # getitem_189
    torch.randn(1605632, dtype=torch.float32, device='cuda').as_strided([4, 512, 28, 28], [401408, 1, 14336, 512]),  # convolution_24
    torch.randn([1, 512, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
