"""
Standalone repro captured via capture_hook.
Label: swin_t_training
Pattern hash: 741c0d7d8d11
Shape hash: c39aed4b
"""
import torch
import torch._inductor.config as inductor_config
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, getitem_8: "f32[4, 56, 56, 1]", add_18: "f32[4, 56, 56, 96]", getitem_9: "f32[4, 56, 56, 1]", primals_28: "f32[96]", primals_29: "f32[96]", _shape_param_0, primals_30: "f32[384, 96]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:454 in forward, code: x = x + self.stochastic_depth(self.mlp(self.norm2(x)))
        add_tensor: "f32[4, 56, 56, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-05);  getitem_8 = None
        rsqrt_default: "f32[4, 56, 56, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        sub_tensor: "f32[4, 56, 56, 96]" = torch.ops.aten.sub.Tensor(add_18, getitem_9);  add_18 = getitem_9 = None
        mul_tensor: "f32[4, 56, 56, 96]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[4, 56, 56, 96]" = torch.ops.aten.mul.Tensor(mul_tensor, primals_28);  mul_tensor = primals_28 = None
        add_tensor_1: "f32[4, 56, 56, 96]" = torch.ops.aten.add.Tensor(mul_tensor_1, primals_29);  mul_tensor_1 = primals_29 = None
        reshape_default: "f32[12544, 96]" = torch.ops.aten.reshape.default(add_tensor_1, _shape_param_0);  add_tensor_1 = _shape_param_0 = None
        permute_default: "f32[96, 384]" = torch.ops.aten.permute.default(primals_30, [1, 0]);  primals_30 = None
        return (reshape_default, permute_default)



def make_inputs():
    return [
    torch.randn([4, 56, 56, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 56, 56, 96], dtype=torch.float32, device='cuda'),
    torch.randn([4, 56, 56, 1], dtype=torch.float32, device='cuda'),
    torch.randn([96], dtype=torch.float32, device='cuda'),
    torch.randn([96], dtype=torch.float32, device='cuda'),
    [12544, 96],  # _shape_param_0
    torch.randn([384, 96], dtype=torch.float32, device='cuda'),
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
