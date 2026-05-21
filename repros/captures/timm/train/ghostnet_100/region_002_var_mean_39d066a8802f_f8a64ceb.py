"""
Standalone repro captured via capture_hook.
Label: timm_ghostnet_100_train
Pattern hash: 39d066a8802f
Shape hash: f8a64ceb
"""
_shapes_config = "(T([512, 80, 7, 7], f32, stride=(3920, 1, 560, 80)), T([80], f32), T([80], f32), T([512, 80, 7, 7], f32, stride=(3920, 1, 560, 80)), T([512, 160, 7, 7], f32, stride=(7840, 1, 1120, 160)))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, convolution_92: "f32[512, 80, 7, 7]", primals_502: "f32[80]", primals_503: "f32[80]", add_411: "f32[512, 80, 7, 7]", add_395: "f32[512, 160, 7, 7]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:69 in forward, code: x2 = self.cheap_operation(x1)
        var_mean_correction = torch.ops.aten.var_mean.correction(convolution_92, [0, 2, 3], correction = 0, keepdim = True)
        getitem: "f32[1, 80, 1, 1]" = var_mean_correction[0]
        getitem_1: "f32[1, 80, 1, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor: "f32[1, 80, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[1, 80, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        sub_tensor: "f32[512, 80, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_92, getitem_1);  convolution_92 = getitem_1 = None
        mul_tensor: "f32[512, 80, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        unsqueeze_default: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(primals_502, -1);  primals_502 = None
        unsqueeze_default_1: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_1: "f32[512, 80, 7, 7]" = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default_1);  mul_tensor = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(primals_503, -1);  primals_503 = None
        unsqueeze_default_3: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor_1: "f32[512, 80, 7, 7]" = torch.ops.aten.add.Tensor(mul_tensor_1, unsqueeze_default_3);  mul_tensor_1 = unsqueeze_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:70 in forward, code: out = torch.cat([x1, x2], dim=1)
        cat_default: "f32[512, 160, 7, 7]" = torch.ops.aten.cat.default([add_411, add_tensor_1], 1);  add_411 = add_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:445 in forward, code: x += self.shortcut(shortcut)
        add_tensor_2: "f32[512, 160, 7, 7]" = torch.ops.aten.add.Tensor(cat_default, add_395);  cat_default = add_395 = None
        return add_tensor_2



def make_inputs():
    return [
    torch.randn(2007040, dtype=torch.float32, device='cuda').as_strided([512, 80, 7, 7], [3920, 1, 560, 80]),  # convolution_92
    torch.randn([80], dtype=torch.float32, device='cuda'),
    torch.randn([80], dtype=torch.float32, device='cuda'),
    torch.randn(2007040, dtype=torch.float32, device='cuda').as_strided([512, 80, 7, 7], [3920, 1, 560, 80]),  # add_411
    torch.randn(4014080, dtype=torch.float32, device='cuda').as_strided([512, 160, 7, 7], [7840, 1, 1120, 160]),  # add_395
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
