"""
Standalone repro captured via capture_hook.
Label: swin_t_training
Pattern hash: 741c0d7d8d11
Shape hash: 620d3428
"""
import torch
import torch._inductor.config as inductor_config
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, getitem_20: "f32[4, 14, 14, 1]", cat_1: "f32[4, 14, 14, 768]", getitem_21: "f32[4, 14, 14, 1]", primals_65: "f32[768]", primals_66: "f32[768]", primals_67: "f32[384, 768]", _shape_param_0):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:84 in forward, code: x = self.norm(x)
        add_tensor: "f32[4, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-05);  getitem_20 = None
        rsqrt_default: "f32[4, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        sub_tensor: "f32[4, 14, 14, 768]" = torch.ops.aten.sub.Tensor(cat_1, getitem_21);  cat_1 = getitem_21 = None
        mul_tensor: "f32[4, 14, 14, 768]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[4, 14, 14, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, primals_65);  mul_tensor = primals_65 = None
        add_tensor_1: "f32[4, 14, 14, 768]" = torch.ops.aten.add.Tensor(mul_tensor_1, primals_66);  mul_tensor_1 = primals_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:85 in forward, code: x = self.reduction(x)  # ... H/2 W/2 2*C
        permute_default: "f32[768, 384]" = torch.ops.aten.permute.default(primals_67, [1, 0]);  primals_67 = None
        reshape_default: "f32[784, 768]" = torch.ops.aten.reshape.default(add_tensor_1, _shape_param_0);  add_tensor_1 = _shape_param_0 = None
        return (permute_default, reshape_default)



def make_inputs():
    return [
    torch.randn([4, 14, 14, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 14, 14, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 14, 14, 1], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([384, 768], dtype=torch.float32, device='cuda'),
    [784, 768],  # _shape_param_0
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
