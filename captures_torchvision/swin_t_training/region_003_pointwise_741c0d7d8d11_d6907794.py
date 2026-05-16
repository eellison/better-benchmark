"""
Standalone repro captured via capture_hook.
Label: swin_t_training
Pattern hash: 741c0d7d8d11
Shape hash: d6907794
"""
import torch
import torch._inductor.config as inductor_config
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, getitem_54: "f32[4, 7, 7, 1]", add_124: "f32[4, 7, 7, 768]", getitem_55: "f32[4, 7, 7, 1]", primals_177: "f32[768]", primals_178: "f32[768]", _shape_param_0, primals_179: "f32[3072, 768]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:454 in forward, code: x = x + self.stochastic_depth(self.mlp(self.norm2(x)))
        add_tensor: "f32[4, 7, 7, 1]" = torch.ops.aten.add.Tensor(getitem_54, 1e-05);  getitem_54 = None
        rsqrt_default: "f32[4, 7, 7, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        sub_tensor: "f32[4, 7, 7, 768]" = torch.ops.aten.sub.Tensor(add_124, getitem_55);  add_124 = getitem_55 = None
        mul_tensor: "f32[4, 7, 7, 768]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[4, 7, 7, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, primals_177);  mul_tensor = primals_177 = None
        add_tensor_1: "f32[4, 7, 7, 768]" = torch.ops.aten.add.Tensor(mul_tensor_1, primals_178);  mul_tensor_1 = primals_178 = None
        reshape_default: "f32[196, 768]" = torch.ops.aten.reshape.default(add_tensor_1, _shape_param_0);  add_tensor_1 = _shape_param_0 = None
        permute_default: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_179, [1, 0]);  primals_179 = None
        return (reshape_default, permute_default)



def make_inputs():
    return [
    torch.randn([4, 7, 7, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 7, 7, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 7, 7, 1], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    [196, 768],  # _shape_param_0
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
