"""
Standalone repro captured via capture_hook.
Label: timm_ghostnet_100_train
Pattern hash: fc989ccddd70
Shape hash: e09b7170
"""
_shapes_config = "(T([512, 92, 14, 14], f32, stride=(18032, 1, 1288, 92)), T([92], f32), T([92], f32), T([512, 92, 14, 14], f32, stride=(18032, 1, 1288, 92)))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, convolution_47: "f32[512, 92, 14, 14]", primals_272: "f32[92]", primals_273: "f32[92]", relu_19: "f32[512, 92, 14, 14]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:69 in forward, code: x2 = self.cheap_operation(x1)
        var_mean_correction = torch.ops.aten.var_mean.correction(convolution_47, [0, 2, 3], correction = 0, keepdim = True)
        getitem: "f32[1, 92, 1, 1]" = var_mean_correction[0]
        getitem_1: "f32[1, 92, 1, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor: "f32[1, 92, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[1, 92, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        sub_tensor: "f32[512, 92, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_47, getitem_1);  convolution_47 = getitem_1 = None
        mul_tensor: "f32[512, 92, 14, 14]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        unsqueeze_default: "f32[92, 1]" = torch.ops.aten.unsqueeze.default(primals_272, -1);  primals_272 = None
        unsqueeze_default_1: "f32[92, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_1: "f32[512, 92, 14, 14]" = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default_1);  mul_tensor = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[92, 1]" = torch.ops.aten.unsqueeze.default(primals_273, -1);  primals_273 = None
        unsqueeze_default_3: "f32[92, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor_1: "f32[512, 92, 14, 14]" = torch.ops.aten.add.Tensor(mul_tensor_1, unsqueeze_default_3);  mul_tensor_1 = unsqueeze_default_3 = None
        relu_default: "f32[512, 92, 14, 14]" = torch.ops.aten.relu.default(add_tensor_1);  add_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:70 in forward, code: out = torch.cat([x1, x2], dim=1)
        cat_default: "f32[512, 184, 14, 14]" = torch.ops.aten.cat.default([relu_19, relu_default], 1);  relu_19 = relu_default = None
        return cat_default



def make_inputs():
    return [
    torch.randn(9232384, dtype=torch.float32, device='cuda').as_strided([512, 92, 14, 14], [18032, 1, 1288, 92]),  # convolution_47
    torch.randn([92], dtype=torch.float32, device='cuda'),
    torch.randn([92], dtype=torch.float32, device='cuda'),
    torch.randn(9232384, dtype=torch.float32, device='cuda').as_strided([512, 92, 14, 14], [18032, 1, 1288, 92]),  # relu_19
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
