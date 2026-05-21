"""
Standalone repro captured via capture_hook.
Label: timm_ghostnet_100_train
Pattern hash: 2096009f00b8
Shape hash: b4d5f01f
"""
_shapes_config = "(T([512, 480, 7, 7], f32, stride=(23520, 1, 3360, 480)), T([480], f32), T([480], f32))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, convolution_87: "f32[512, 480, 7, 7]", primals_480: "f32[480]", primals_481: "f32[480]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:68 in forward, code: x1 = self.primary_conv(x)
        var_mean_correction = torch.ops.aten.var_mean.correction(convolution_87, [0, 2, 3], correction = 0, keepdim = True)
        getitem: "f32[1, 480, 1, 1]" = var_mean_correction[0]
        getitem_1: "f32[1, 480, 1, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor: "f32[1, 480, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[1, 480, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        sub_tensor: "f32[512, 480, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_87, getitem_1);  convolution_87 = getitem_1 = None
        mul_tensor: "f32[512, 480, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        unsqueeze_default: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_480, -1);  primals_480 = None
        unsqueeze_default_1: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_1: "f32[512, 480, 7, 7]" = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default_1);  mul_tensor = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_481, -1);  primals_481 = None
        unsqueeze_default_3: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor_1: "f32[512, 480, 7, 7]" = torch.ops.aten.add.Tensor(mul_tensor_1, unsqueeze_default_3);  mul_tensor_1 = unsqueeze_default_3 = None
        relu_default: "f32[512, 480, 7, 7]" = torch.ops.aten.relu.default(add_tensor_1);  add_tensor_1 = None
        return relu_default



def make_inputs():
    return [
    torch.randn(12042240, dtype=torch.float32, device='cuda').as_strided([512, 480, 7, 7], [23520, 1, 3360, 480]),  # convolution_87
    torch.randn([480], dtype=torch.float32, device='cuda'),
    torch.randn([480], dtype=torch.float32, device='cuda'),
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
