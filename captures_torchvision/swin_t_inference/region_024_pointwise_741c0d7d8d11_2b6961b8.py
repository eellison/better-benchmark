"""
Standalone repro captured via capture_hook.
Label: swin_t_inference
Pattern hash: 741c0d7d8d11
Shape hash: 2b6961b8
"""
import torch
import torch._inductor.config as inductor_config
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, cat_1: "f32[1, 14, 14, 768]", getitem_21: "f32[1, 14, 14, 1]", getitem_20: "f32[1, 14, 14, 1]", arg64_1: "f32[768]", arg65_1: "f32[768]", _shape_param_0, arg66_1: "f32[384, 768]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:84 in forward, code: x = self.norm(x)
        sub_tensor: "f32[1, 14, 14, 768]" = torch.ops.aten.sub.Tensor(cat_1, getitem_21);  cat_1 = getitem_21 = None
        add_tensor: "f32[1, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-05);  getitem_20 = None
        rsqrt_default: "f32[1, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        mul_tensor: "f32[1, 14, 14, 768]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[1, 14, 14, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, arg64_1);  mul_tensor = arg64_1 = None
        add_tensor_1: "f32[1, 14, 14, 768]" = torch.ops.aten.add.Tensor(mul_tensor_1, arg65_1);  mul_tensor_1 = arg65_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:85 in forward, code: x = self.reduction(x)  # ... H/2 W/2 2*C
        reshape_default: "f32[196, 768]" = torch.ops.aten.reshape.default(add_tensor_1, _shape_param_0);  add_tensor_1 = _shape_param_0 = None
        permute_default: "f32[768, 384]" = torch.ops.aten.permute.default(arg66_1, [1, 0]);  arg66_1 = None
        return (reshape_default, permute_default)



def make_inputs():
    return [
    torch.randn([1, 14, 14, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1, 14, 14, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 14, 14, 1], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    [196, 768],  # _shape_param_0
    torch.randn([384, 768], dtype=torch.float32, device='cuda'),
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
