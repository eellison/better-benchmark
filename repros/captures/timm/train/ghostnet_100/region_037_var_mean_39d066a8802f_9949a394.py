"""
Standalone repro captured via capture_hook.
Label: timm_ghostnet_100_train
Pattern hash: 39d066a8802f
Shape hash: 9949a394
"""
_shapes_config = "(T([512, 20, 28, 28], f32, stride=(15680, 1, 560, 20)), T([20], f32), T([20], f32), T([512, 20, 28, 28], f32, stride=(15680, 1, 560, 20)), T([512, 40, 28, 28], f32, stride=(31360, 1, 1120, 40)))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, convolution_30: "f32[512, 20, 28, 28]", primals_170: "f32[20]", primals_171: "f32[20]", add_135: "f32[512, 20, 28, 28]", add_119: "f32[512, 40, 28, 28]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:69 in forward, code: x2 = self.cheap_operation(x1)
        var_mean_correction = torch.ops.aten.var_mean.correction(convolution_30, [0, 2, 3], correction = 0, keepdim = True)
        getitem: "f32[1, 20, 1, 1]" = var_mean_correction[0]
        getitem_1: "f32[1, 20, 1, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor: "f32[1, 20, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[1, 20, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        sub_tensor: "f32[512, 20, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_30, getitem_1);  convolution_30 = getitem_1 = None
        mul_tensor: "f32[512, 20, 28, 28]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        unsqueeze_default: "f32[20, 1]" = torch.ops.aten.unsqueeze.default(primals_170, -1);  primals_170 = None
        unsqueeze_default_1: "f32[20, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_1: "f32[512, 20, 28, 28]" = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default_1);  mul_tensor = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[20, 1]" = torch.ops.aten.unsqueeze.default(primals_171, -1);  primals_171 = None
        unsqueeze_default_3: "f32[20, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor_1: "f32[512, 20, 28, 28]" = torch.ops.aten.add.Tensor(mul_tensor_1, unsqueeze_default_3);  mul_tensor_1 = unsqueeze_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:70 in forward, code: out = torch.cat([x1, x2], dim=1)
        cat_default: "f32[512, 40, 28, 28]" = torch.ops.aten.cat.default([add_135, add_tensor_1], 1);  add_135 = add_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:445 in forward, code: x += self.shortcut(shortcut)
        add_tensor_2: "f32[512, 40, 28, 28]" = torch.ops.aten.add.Tensor(cat_default, add_119);  cat_default = add_119 = None
        return add_tensor_2



def make_inputs():
    return [
    torch.randn(8028160, dtype=torch.float32, device='cuda').as_strided([512, 20, 28, 28], [15680, 1, 560, 20]),  # convolution_30
    torch.randn([20], dtype=torch.float32, device='cuda'),
    torch.randn([20], dtype=torch.float32, device='cuda'),
    torch.randn(8028160, dtype=torch.float32, device='cuda').as_strided([512, 20, 28, 28], [15680, 1, 560, 20]),  # add_135
    torch.randn(16056320, dtype=torch.float32, device='cuda').as_strided([512, 40, 28, 28], [31360, 1, 1120, 40]),  # add_119
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
