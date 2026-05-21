"""
Standalone repro captured via capture_hook.
Label: timm_ghostnet_100_train
Pattern hash: 2096009f00b8
Shape hash: b9ff5d6e
"""
_shapes_config = "(T([512, 92, 14, 14], f32, stride=(18032, 1, 1288, 92)), T([92], f32), T([92], f32))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, convolution_46: "f32[512, 92, 14, 14]", primals_266: "f32[92]", primals_267: "f32[92]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:68 in forward, code: x1 = self.primary_conv(x)
        var_mean_correction = torch.ops.aten.var_mean.correction(convolution_46, [0, 2, 3], correction = 0, keepdim = True)
        getitem: "f32[1, 92, 1, 1]" = var_mean_correction[0]
        getitem_1: "f32[1, 92, 1, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor: "f32[1, 92, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[1, 92, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        sub_tensor: "f32[512, 92, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_46, getitem_1);  convolution_46 = getitem_1 = None
        mul_tensor: "f32[512, 92, 14, 14]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        unsqueeze_default: "f32[92, 1]" = torch.ops.aten.unsqueeze.default(primals_266, -1);  primals_266 = None
        unsqueeze_default_1: "f32[92, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_1: "f32[512, 92, 14, 14]" = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default_1);  mul_tensor = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[92, 1]" = torch.ops.aten.unsqueeze.default(primals_267, -1);  primals_267 = None
        unsqueeze_default_3: "f32[92, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor_1: "f32[512, 92, 14, 14]" = torch.ops.aten.add.Tensor(mul_tensor_1, unsqueeze_default_3);  mul_tensor_1 = unsqueeze_default_3 = None
        relu_default: "f32[512, 92, 14, 14]" = torch.ops.aten.relu.default(add_tensor_1);  add_tensor_1 = None
        return relu_default



def make_inputs():
    return [
    torch.randn(9232384, dtype=torch.float32, device='cuda').as_strided([512, 92, 14, 14], [18032, 1, 1288, 92]),  # convolution_46
    torch.randn([92], dtype=torch.float32, device='cuda'),
    torch.randn([92], dtype=torch.float32, device='cuda'),
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
