"""
Standalone repro captured via capture_hook.
Label: vit_b_16_inference
Pattern hash: 1ecc9387d6bd
Shape hash: bae09006
"""
import torch
import torch._inductor.config as inductor_config
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, add_77: "f32[1, 197, 768]", getitem_89: "f32[1, 197, 1]", getitem_88: "f32[1, 197, 1]", arg137_1: "f32[768]", arg138_1: "f32[768]", _shape_param_0, arg140_1: "f32[2304, 768]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vision_transformer.py:112 in forward, code: x = self.ln_1(input)
        sub_tensor: "f32[1, 197, 768]" = torch.ops.aten.sub.Tensor(add_77, getitem_89);  add_77 = getitem_89 = None
        add_tensor: "f32[1, 197, 1]" = torch.ops.aten.add.Tensor(getitem_88, 1e-06);  getitem_88 = None
        rsqrt_default: "f32[1, 197, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        mul_tensor: "f32[1, 197, 768]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[1, 197, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, arg137_1);  mul_tensor = arg137_1 = None
        add_tensor_1: "f32[1, 197, 768]" = torch.ops.aten.add.Tensor(mul_tensor_1, arg138_1);  mul_tensor_1 = arg138_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vision_transformer.py:113 in forward, code: x, _ = self.self_attention(x, x, x, need_weights=False)
        permute_default: "f32[197, 1, 768]" = torch.ops.aten.permute.default(add_tensor_1, [1, 0, 2]);  add_tensor_1 = None
        reshape_default: "f32[197, 768]" = torch.ops.aten.reshape.default(permute_default, _shape_param_0);  permute_default = _shape_param_0 = None
        permute_default_1: "f32[768, 2304]" = torch.ops.aten.permute.default(arg140_1, [1, 0]);  arg140_1 = None
        return (reshape_default, permute_default_1)



def make_inputs():
    return [
    torch.randn([1, 197, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1, 197, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 197, 1], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    [197, 768],  # _shape_param_0
    torch.randn([2304, 768], dtype=torch.float32, device='cuda'),
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
