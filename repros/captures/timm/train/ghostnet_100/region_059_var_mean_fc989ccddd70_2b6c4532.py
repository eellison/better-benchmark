"""
Standalone repro captured via capture_hook.
Label: timm_ghostnet_100_train
Pattern hash: fc989ccddd70
Shape hash: 2b6c4532
"""
_shapes_config = "(T([512, 8, 112, 112], f32, stride=(100352, 1, 896, 8)), T([8], f32), T([8], f32), T([512, 8, 112, 112], f32, stride=(100352, 1, 896, 8)))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, convolution_2: "f32[512, 8, 112, 112]", primals_18: "f32[8]", primals_19: "f32[8]", relu_1: "f32[512, 8, 112, 112]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:69 in forward, code: x2 = self.cheap_operation(x1)
        var_mean_correction = torch.ops.aten.var_mean.correction(convolution_2, [0, 2, 3], correction = 0, keepdim = True)
        getitem: "f32[1, 8, 1, 1]" = var_mean_correction[0]
        getitem_1: "f32[1, 8, 1, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor: "f32[1, 8, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[1, 8, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        sub_tensor: "f32[512, 8, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_2, getitem_1);  convolution_2 = getitem_1 = None
        mul_tensor: "f32[512, 8, 112, 112]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        unsqueeze_default: "f32[8, 1]" = torch.ops.aten.unsqueeze.default(primals_18, -1);  primals_18 = None
        unsqueeze_default_1: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_1: "f32[512, 8, 112, 112]" = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default_1);  mul_tensor = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[8, 1]" = torch.ops.aten.unsqueeze.default(primals_19, -1);  primals_19 = None
        unsqueeze_default_3: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor_1: "f32[512, 8, 112, 112]" = torch.ops.aten.add.Tensor(mul_tensor_1, unsqueeze_default_3);  mul_tensor_1 = unsqueeze_default_3 = None
        relu_default: "f32[512, 8, 112, 112]" = torch.ops.aten.relu.default(add_tensor_1);  add_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:70 in forward, code: out = torch.cat([x1, x2], dim=1)
        cat_default: "f32[512, 16, 112, 112]" = torch.ops.aten.cat.default([relu_1, relu_default], 1);  relu_1 = relu_default = None
        return cat_default



def make_inputs():
    return [
    torch.randn(51380224, dtype=torch.float32, device='cuda').as_strided([512, 8, 112, 112], [100352, 1, 896, 8]),  # convolution_2
    torch.randn([8], dtype=torch.float32, device='cuda'),
    torch.randn([8], dtype=torch.float32, device='cuda'),
    torch.randn(51380224, dtype=torch.float32, device='cuda').as_strided([512, 8, 112, 112], [100352, 1, 896, 8]),  # relu_1
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
