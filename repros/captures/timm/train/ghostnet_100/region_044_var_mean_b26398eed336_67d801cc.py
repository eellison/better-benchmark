"""
Standalone repro captured via capture_hook.
Label: timm_ghostnet_100_train
Pattern hash: b26398eed336
Shape hash: 67d801cc
"""
_shapes_config = "(T([512, 20, 28, 28], f32, stride=(15680, 1, 560, 20)), T([20], f32), T([20], f32), T([512, 24, 28, 28], f32, stride=(18816, 1, 672, 24)), T([24], f32), T([24], f32))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, convolution_21: "f32[512, 20, 28, 28]", primals_124: "f32[20]", primals_125: "f32[20]", convolution_23: "f32[512, 24, 28, 28]", primals_136: "f32[24]", primals_137: "f32[24]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:68 in forward, code: x1 = self.primary_conv(x)
        var_mean_correction = torch.ops.aten.var_mean.correction(convolution_21, [0, 2, 3], correction = 0, keepdim = True)
        getitem: "f32[1, 20, 1, 1]" = var_mean_correction[0]
        getitem_1: "f32[1, 20, 1, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor: "f32[1, 20, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[1, 20, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        sub_tensor: "f32[512, 20, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_21, getitem_1);  convolution_21 = getitem_1 = None
        mul_tensor: "f32[512, 20, 28, 28]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        unsqueeze_default: "f32[20, 1]" = torch.ops.aten.unsqueeze.default(primals_124, -1);  primals_124 = None
        unsqueeze_default_1: "f32[20, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_1: "f32[512, 20, 28, 28]" = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default_1);  mul_tensor = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[20, 1]" = torch.ops.aten.unsqueeze.default(primals_125, -1);  primals_125 = None
        unsqueeze_default_3: "f32[20, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor_1: "f32[512, 20, 28, 28]" = torch.ops.aten.add.Tensor(mul_tensor_1, unsqueeze_default_3);  mul_tensor_1 = unsqueeze_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:445 in forward, code: x += self.shortcut(shortcut)
        var_mean_correction_1 = torch.ops.aten.var_mean.correction(convolution_23, [0, 2, 3], correction = 0, keepdim = True)
        getitem_2: "f32[1, 24, 1, 1]" = var_mean_correction_1[0]
        getitem_3: "f32[1, 24, 1, 1]" = var_mean_correction_1[1];  var_mean_correction_1 = None
        add_tensor_2: "f32[1, 24, 1, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-05);  getitem_2 = None
        rsqrt_default_1: "f32[1, 24, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor_2);  add_tensor_2 = None
        sub_tensor_1: "f32[512, 24, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_23, getitem_3);  convolution_23 = getitem_3 = None
        mul_tensor_2: "f32[512, 24, 28, 28]" = torch.ops.aten.mul.Tensor(sub_tensor_1, rsqrt_default_1);  sub_tensor_1 = rsqrt_default_1 = None
        unsqueeze_default_4: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(primals_136, -1);  primals_136 = None
        unsqueeze_default_5: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, -1);  unsqueeze_default_4 = None
        mul_tensor_3: "f32[512, 24, 28, 28]" = torch.ops.aten.mul.Tensor(mul_tensor_2, unsqueeze_default_5);  mul_tensor_2 = unsqueeze_default_5 = None
        unsqueeze_default_6: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(primals_137, -1);  primals_137 = None
        unsqueeze_default_7: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, -1);  unsqueeze_default_6 = None
        add_tensor_3: "f32[512, 24, 28, 28]" = torch.ops.aten.add.Tensor(mul_tensor_3, unsqueeze_default_7);  mul_tensor_3 = unsqueeze_default_7 = None
        return (add_tensor_1, add_tensor_3)



def make_inputs():
    return [
    torch.randn(8028160, dtype=torch.float32, device='cuda').as_strided([512, 20, 28, 28], [15680, 1, 560, 20]),  # convolution_21
    torch.randn([20], dtype=torch.float32, device='cuda'),
    torch.randn([20], dtype=torch.float32, device='cuda'),
    torch.randn(9633792, dtype=torch.float32, device='cuda').as_strided([512, 24, 28, 28], [18816, 1, 672, 24]),  # convolution_23
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
