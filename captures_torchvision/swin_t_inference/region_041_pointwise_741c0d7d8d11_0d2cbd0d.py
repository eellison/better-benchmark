"""
Standalone repro captured via capture_hook.
Label: swin_t_inference
Pattern hash: 741c0d7d8d11
Shape hash: 0d2cbd0d
"""
import torch
import torch._inductor.config as inductor_config
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, add_18: "f32[1, 56, 56, 96]", getitem_9: "f32[1, 56, 56, 1]", getitem_8: "f32[1, 56, 56, 1]", arg27_1: "f32[96]", arg28_1: "f32[96]", _shape_param_0, arg29_1: "f32[384, 96]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:454 in forward, code: x = x + self.stochastic_depth(self.mlp(self.norm2(x)))
        sub_tensor: "f32[1, 56, 56, 96]" = torch.ops.aten.sub.Tensor(add_18, getitem_9);  add_18 = getitem_9 = None
        add_tensor: "f32[1, 56, 56, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-05);  getitem_8 = None
        rsqrt_default: "f32[1, 56, 56, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        mul_tensor: "f32[1, 56, 56, 96]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[1, 56, 56, 96]" = torch.ops.aten.mul.Tensor(mul_tensor, arg27_1);  mul_tensor = arg27_1 = None
        add_tensor_1: "f32[1, 56, 56, 96]" = torch.ops.aten.add.Tensor(mul_tensor_1, arg28_1);  mul_tensor_1 = arg28_1 = None
        reshape_default: "f32[3136, 96]" = torch.ops.aten.reshape.default(add_tensor_1, _shape_param_0);  add_tensor_1 = _shape_param_0 = None
        permute_default: "f32[96, 384]" = torch.ops.aten.permute.default(arg29_1, [1, 0]);  arg29_1 = None
        return (reshape_default, permute_default)



def make_inputs():
    return [
    torch.randn([1, 56, 56, 96], dtype=torch.float32, device='cuda'),
    torch.randn([1, 56, 56, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 56, 56, 1], dtype=torch.float32, device='cuda'),
    torch.randn([96], dtype=torch.float32, device='cuda'),
    torch.randn([96], dtype=torch.float32, device='cuda'),
    [3136, 96],  # _shape_param_0
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
