"""
Standalone repro captured via capture_hook.
Label: vit_b_16_inference
Pattern hash: 741c0d7d8d11
Shape hash: 069ebc38
"""
import torch
import torch._inductor.config as inductor_config
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, add_80: "f32[1, 197, 768]", getitem_95: "f32[1, 197, 1]", getitem_94: "f32[1, 197, 1]", arg143_1: "f32[768]", arg144_1: "f32[768]", _shape_param_0, arg145_1: "f32[3072, 768]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vision_transformer.py:117 in forward, code: y = self.ln_2(x)
        sub_tensor: "f32[1, 197, 768]" = torch.ops.aten.sub.Tensor(add_80, getitem_95);  add_80 = getitem_95 = None
        add_tensor: "f32[1, 197, 1]" = torch.ops.aten.add.Tensor(getitem_94, 1e-06);  getitem_94 = None
        rsqrt_default: "f32[1, 197, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        mul_tensor: "f32[1, 197, 768]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[1, 197, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, arg143_1);  mul_tensor = arg143_1 = None
        add_tensor_1: "f32[1, 197, 768]" = torch.ops.aten.add.Tensor(mul_tensor_1, arg144_1);  mul_tensor_1 = arg144_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vision_transformer.py:118 in forward, code: y = self.mlp(y)
        reshape_default: "f32[197, 768]" = torch.ops.aten.reshape.default(add_tensor_1, _shape_param_0);  add_tensor_1 = _shape_param_0 = None
        permute_default: "f32[768, 3072]" = torch.ops.aten.permute.default(arg145_1, [1, 0]);  arg145_1 = None
        return (reshape_default, permute_default)



def make_inputs():
    return [
    torch.randn([1, 197, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1, 197, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 197, 1], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    [197, 768],  # _shape_param_0
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
