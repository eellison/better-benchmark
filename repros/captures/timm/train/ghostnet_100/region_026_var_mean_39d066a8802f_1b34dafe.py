"""
Standalone repro captured via capture_hook.
Label: timm_ghostnet_100_train
Pattern hash: 39d066a8802f
Shape hash: 1b34dafe
"""
_shapes_config = "(T([512, 40, 14, 14], f32, stride=(7840, 1, 560, 40)), T([40], f32), T([40], f32), T([512, 40, 14, 14], f32, stride=(7840, 1, 560, 40)), T([512, 80, 14, 14], f32, stride=(15680, 1, 1120, 80)))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, convolution_49: "f32[512, 40, 14, 14]", primals_284: "f32[40]", primals_285: "f32[40]", add_234: "f32[512, 40, 14, 14]", add_219: "f32[512, 80, 14, 14]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:69 in forward, code: x2 = self.cheap_operation(x1)
        var_mean_correction = torch.ops.aten.var_mean.correction(convolution_49, [0, 2, 3], correction = 0, keepdim = True)
        getitem: "f32[1, 40, 1, 1]" = var_mean_correction[0]
        getitem_1: "f32[1, 40, 1, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor: "f32[1, 40, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[1, 40, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        sub_tensor: "f32[512, 40, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_49, getitem_1);  convolution_49 = getitem_1 = None
        mul_tensor: "f32[512, 40, 14, 14]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        unsqueeze_default: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(primals_284, -1);  primals_284 = None
        unsqueeze_default_1: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_1: "f32[512, 40, 14, 14]" = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default_1);  mul_tensor = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(primals_285, -1);  primals_285 = None
        unsqueeze_default_3: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor_1: "f32[512, 40, 14, 14]" = torch.ops.aten.add.Tensor(mul_tensor_1, unsqueeze_default_3);  mul_tensor_1 = unsqueeze_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:70 in forward, code: out = torch.cat([x1, x2], dim=1)
        cat_default: "f32[512, 80, 14, 14]" = torch.ops.aten.cat.default([add_234, add_tensor_1], 1);  add_234 = add_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:445 in forward, code: x += self.shortcut(shortcut)
        add_tensor_2: "f32[512, 80, 14, 14]" = torch.ops.aten.add.Tensor(cat_default, add_219);  cat_default = add_219 = None
        return add_tensor_2



def make_inputs():
    return [
    torch.randn(4014080, dtype=torch.float32, device='cuda').as_strided([512, 40, 14, 14], [7840, 1, 560, 40]),  # convolution_49
    torch.randn([40], dtype=torch.float32, device='cuda'),
    torch.randn([40], dtype=torch.float32, device='cuda'),
    torch.randn(4014080, dtype=torch.float32, device='cuda').as_strided([512, 40, 14, 14], [7840, 1, 560, 40]),  # add_234
    torch.randn(8028160, dtype=torch.float32, device='cuda').as_strided([512, 80, 14, 14], [15680, 1, 1120, 80]),  # add_219
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
