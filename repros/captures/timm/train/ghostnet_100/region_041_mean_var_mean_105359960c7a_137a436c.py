"""
Standalone repro captured via capture_hook.
Label: timm_ghostnet_100_train
Pattern hash: 105359960c7a
Shape hash: 137a436c
"""
_shapes_config = "(T([512, 60, 28, 28], f32, stride=(47040, 1, 1680, 60)), T([60], f32), T([60], f32), T([512, 60, 28, 28], f32, stride=(47040, 1, 1680, 60)))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, convolution_26: "f32[512, 60, 28, 28]", primals_154: "f32[60]", primals_155: "f32[60]", relu_10: "f32[512, 60, 28, 28]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:69 in forward, code: x2 = self.cheap_operation(x1)
        var_mean_correction = torch.ops.aten.var_mean.correction(convolution_26, [0, 2, 3], correction = 0, keepdim = True)
        getitem: "f32[1, 60, 1, 1]" = var_mean_correction[0]
        getitem_1: "f32[1, 60, 1, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor: "f32[1, 60, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[1, 60, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        sub_tensor: "f32[512, 60, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_26, getitem_1);  convolution_26 = getitem_1 = None
        mul_tensor: "f32[512, 60, 28, 28]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        unsqueeze_default: "f32[60, 1]" = torch.ops.aten.unsqueeze.default(primals_154, -1);  primals_154 = None
        unsqueeze_default_1: "f32[60, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_1: "f32[512, 60, 28, 28]" = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default_1);  mul_tensor = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[60, 1]" = torch.ops.aten.unsqueeze.default(primals_155, -1);  primals_155 = None
        unsqueeze_default_3: "f32[60, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor_1: "f32[512, 60, 28, 28]" = torch.ops.aten.add.Tensor(mul_tensor_1, unsqueeze_default_3);  mul_tensor_1 = unsqueeze_default_3 = None
        relu_default: "f32[512, 60, 28, 28]" = torch.ops.aten.relu.default(add_tensor_1);  add_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:70 in forward, code: out = torch.cat([x1, x2], dim=1)
        cat_default: "f32[512, 120, 28, 28]" = torch.ops.aten.cat.default([relu_10, relu_default], 1);  relu_10 = relu_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:79 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_dim: "f32[512, 120, 1, 1]" = torch.ops.aten.mean.dim(cat_default, [2, 3], True);  cat_default = None
        return mean_dim



def make_inputs():
    return [
    torch.randn(24084480, dtype=torch.float32, device='cuda').as_strided([512, 60, 28, 28], [47040, 1, 1680, 60]),  # convolution_26
    torch.randn([60], dtype=torch.float32, device='cuda'),
    torch.randn([60], dtype=torch.float32, device='cuda'),
    torch.randn(24084480, dtype=torch.float32, device='cuda').as_strided([512, 60, 28, 28], [47040, 1, 1680, 60]),  # relu_10
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
